//! Building an `Index` to efficiently map vocabulary tokens to state transitions.

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::error::{CompileError, CompileStage};
use crate::grammar::CompiledByteDfa;
use crate::prelude::*;
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};

/// `Index` efficiently maps vocabulary tokens to state transitions.
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Index {
    /// The ID of the initial state in the automaton, processing begins from this state.
    initial_state: StateId,
    /// A collection of states considered as terminal states.
    final_states: HashSet<StateId>,
    /// A mapping of state transitions, defined by tokens ids and their corresponding state changes.
    ///
    /// ### Example
    /// ```ignore
    /// transitions = {
    ///    1: {10: 2, 15: 3},
    ///    2: {20: 4, 25: 3},
    ///    3: {30: 4},
    ///    4: {40: 4},
    /// }
    ///  +--------------------------------------+
    ///  |               State 1                |
    ///  |            Initial State             |
    ///  +--------------------------------------+
    ///              |                     |
    ///              +                     |
    ///         Token ID 10                |
    ///  +-----------------------+         |
    ///  |        State 2        |         |
    ///  +-----------------------+         |
    ///       |             |              |
    ///       |             +              +
    ///       |        Token ID 25    Token ID 15
    ///       |        +------------------------+
    ///       |        |        State 3         |
    ///       |        +------------------------+
    ///       |                            |
    ///       +                            +
    ///  Token ID 20                  Token ID 30
    ///  +--------------------------------------+
    ///  |               State 4                |
    ///  |             Final state              |
    ///  +--------------------------------------+
    /// ```
    transitions: HashMap<StateId, HashMap<TokenId, StateId>>,
    /// The token ID reserved for the "end-of-sequence" token.
    eos_token_id: TokenId,
    /// The size of the vocabulary used to build the index.
    vocab_size: usize,
}
/// The `Index` structure is designed to efficiently map tokens from a given vocabulary
/// to state transitions within a finite-state automaton.
///
/// ## Usage:
/// The `Index` is typically constructed by combining a vocabulary and regular expressions.
/// Once built, it can be used to efficiently evaluate token sequences or to validate input data.
///
/// ## Example:
/// ```rust
/// use oc_earley::prelude::*;
///
/// # fn run() -> Result<(), oc_earley::Error> {
/// let regex = "0|[1-9][0-9]*";
/// let mut vocabulary = Vocabulary::new(2);
/// vocabulary.try_insert("0", 0)?;
/// vocabulary.try_insert("1", 1)?;
/// let index = Index::new(regex, &vocabulary)?;
///
/// let initial_state = index.initial_state();
/// println!("Initial state is {}", initial_state);
/// println!("Is initial state a final state? {}", index.is_final_state(&initial_state));
///
/// let allowed_tokens = index.allowed_tokens(&initial_state).expect("Some allowed tokens");
/// println!("Allowed tokens at initial state are {:?}", allowed_tokens);
///
/// let token_id = allowed_tokens.first().expect("First token");
/// println!("Next state for the token_id {} is {:?}", token_id, index.next_state(&initial_state, token_id));
///
/// println!("Final states are {:?}", index.final_states());
/// println!("Index has exactly {} transitions", index.transitions().len());
/// # Ok(())
/// # }
///
/// ```
///
/// ## Performance:
/// - **Complexity**:
///   The `Index` can accommodate large vocabularies and complex regular expressions.
///   However, its size may grow significantly with the complexity of the input.
/// - **Construction Cost**:
///   Building the `Index` involves processing the vocabulary and regular expressions,
///   which may require a considerable amount of time and computational resources.
impl Index {
    /// Builds an `Index` from regular expression and vocabulary tokens.
    pub fn new(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
        let dfa = DFA::new(regex).map_err(Box::new)?;
        Self::project_byte_dfa(&dfa, vocabulary, ProjectionPolicy::Legacy { regex })
    }

    pub(crate) fn from_certified_dfa(
        dfa: &CompiledByteDfa,
        vocabulary: &Vocabulary,
    ) -> Result<Self> {
        Self::project_byte_dfa(dfa, vocabulary, ProjectionPolicy::Exact)
    }

    fn project_byte_dfa<D: ByteDfa>(
        dfa: &D,
        vocabulary: &Vocabulary,
        policy: ProjectionPolicy<'_>,
    ) -> Result<Self> {
        let vocab_size = vocabulary_width(vocabulary)?;
        let eos_token_id = vocabulary.eos_token_id();
        let start_state = dfa.start_state().ok_or(Error::DfaHasNoStartState)?;
        let stable_start = dfa.stable_state_id(start_state);

        if matches!(policy, ProjectionPolicy::Exact)
            && !dfa.is_live(start_state)
            && !dfa.is_accepting(start_state)
        {
            return Ok(Self {
                initial_state: stable_start,
                final_states: HashSet::default(),
                transitions: HashMap::default(),
                eos_token_id,
                vocab_size,
            });
        }

        let mut transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        let mut final_states: HashSet<StateId> = HashSet::default();

        let mut seen: HashSet<D::State> = HashSet::from_iter([start_state]);
        let mut next_states = vec![start_state];
        let mut live_cache: HashMap<D::State, bool> = HashMap::default();

        while let Some(current_state) = next_states.pop() {
            let mut has_valid_transitions = false;
            let stable_current = dfa.stable_state_id(current_state);

            if dfa.is_accepting(current_state) {
                final_states.insert(stable_current);
                has_valid_transitions = true;
            }

            'token_loop: for (token, ids) in vocabulary.tokens().iter() {
                let mut next_state = current_state;
                for transition_byte in token {
                    next_state = dfa.next_state(next_state, *transition_byte);
                    if dfa.is_dead(next_state) {
                        continue 'token_loop;
                    }
                }

                let is_live = *live_cache
                    .entry(next_state)
                    .or_insert_with(|| dfa.is_live(next_state));
                if is_live {
                    has_valid_transitions = true;
                    let stable_next = dfa.stable_state_id(next_state);
                    for token_id in ids {
                        let row = transitions.entry(stable_current).or_default();
                        if row
                            .get(token_id)
                            .is_some_and(|existing| *existing != stable_next)
                        {
                            return Err(Error::AmbiguousTokenId {
                                token_id: *token_id,
                            });
                        }
                        row.insert(*token_id, stable_next);
                    }
                    if seen.insert(next_state) {
                        next_states.push(next_state);
                    }
                }
            }

            if !has_valid_transitions && !dfa.is_accepting(current_state) {
                let mut valid_characters = Vec::new();
                for byte in 0..=255u8 {
                    let test_state = dfa.next_state(current_state, byte);
                    let report_byte = match policy {
                        ProjectionPolicy::Legacy { .. } => !dfa.is_dead(test_state),
                        ProjectionPolicy::Exact => {
                            !dfa.is_dead(test_state) && dfa.is_live(test_state)
                        }
                    };
                    if report_byte {
                        if byte.is_ascii() {
                            valid_characters.push(char::from(byte).to_string());
                        } else {
                            valid_characters.push(format!("\\x{:02x}", byte));
                        }
                    }
                }

                return Err(Error::IncompatibleVocabulary {
                    regex: policy.label().to_owned(),
                    error_state: stable_current,
                    missing_tokens: valid_characters,
                });
            }
        }

        // Populate `transitions` with mappings from `final_states` to `eos_token_id`
        for &final_state in &final_states {
            transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
        }

        Ok(Self {
            initial_state: stable_start,
            final_states,
            transitions,
            eos_token_id,
            vocab_size,
        })
    }

    /// Returns the ID of the initial state in the automaton.
    pub fn initial_state(&self) -> StateId {
        self.initial_state
    }

    /// Returns set of final states.
    pub fn final_states(&self) -> &HashSet<StateId> {
        &self.final_states
    }

    /// Returns state transitions map of tokens ids and their corresponding transition states.
    pub fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        &self.transitions
    }

    /// Checks if state is in final states set or not.
    pub fn is_final_state(&self, state: &StateId) -> bool {
        self.final_states.contains(state)
    }

    /// Lists allowed tokens for a give state ID or `None` if it is not found in `Index`.
    pub fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        self.transitions
            .get(state)
            .map(|transitions| transitions.keys().copied().collect())
    }

    pub fn allowed_tokens_iter(&self, state: &StateId) -> Option<impl Iterator<Item = &TokenId>> {
        self.transitions.get(state).map(|map| map.keys())
    }

    /// Returns transition state for a given state and token id or `None` otherwise.
    pub fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if token_id == &self.eos_token_id {
            return None;
        }
        Some(*self.transitions.get(state)?.get(token_id)?)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

trait ByteDfa {
    type State: Copy + Eq + std::hash::Hash;

    fn start_state(&self) -> Option<Self::State>;
    fn next_state(&self, state: Self::State, byte: u8) -> Self::State;
    fn is_dead(&self, state: Self::State) -> bool;
    fn is_accepting(&self, state: Self::State) -> bool;
    fn is_live(&self, state: Self::State) -> bool;
    fn stable_state_id(&self, state: Self::State) -> StateId;
}

impl ByteDfa for DFA<Vec<u32>> {
    type State = AutomataStateId;

    fn start_state(&self) -> Option<Self::State> {
        self.universal_start_state(Anchored::Yes)
    }

    fn next_state(&self, state: Self::State, byte: u8) -> Self::State {
        Automaton::next_state(self, state, byte)
    }

    fn is_dead(&self, state: Self::State) -> bool {
        self.is_dead_state(state) || self.is_quit_state(state)
    }

    fn is_accepting(&self, state: Self::State) -> bool {
        self.is_match_state(self.next_eoi_state(state))
    }

    fn is_live(&self, state: Self::State) -> bool {
        self.is_accepting(state)
            || self
                .byte_classes()
                .representatives(..)
                .any(|representative| {
                    representative.as_u8().is_some_and(|byte| {
                        let next = Automaton::next_state(self, state, byte);
                        !self.is_dead_state(next) && !self.is_quit_state(next)
                    })
                })
    }

    fn stable_state_id(&self, state: Self::State) -> StateId {
        state.as_u32()
    }
}

impl ByteDfa for CompiledByteDfa {
    type State = StateId;

    fn start_state(&self) -> Option<Self::State> {
        Some(CompiledByteDfa::start_state(self))
    }

    fn next_state(&self, state: Self::State, byte: u8) -> Self::State {
        CompiledByteDfa::next_state(self, state, byte)
    }

    fn is_dead(&self, state: Self::State) -> bool {
        CompiledByteDfa::is_dead(self, state)
    }

    fn is_accepting(&self, state: Self::State) -> bool {
        CompiledByteDfa::is_accepting(self, state)
    }

    fn is_live(&self, state: Self::State) -> bool {
        CompiledByteDfa::is_live(self, state)
    }

    fn stable_state_id(&self, state: Self::State) -> StateId {
        state
    }
}

#[derive(Clone, Copy)]
enum ProjectionPolicy<'a> {
    Legacy { regex: &'a str },
    Exact,
}

impl<'a> ProjectionPolicy<'a> {
    fn label(self) -> &'a str {
        match self {
            Self::Legacy { regex } => regex,
            Self::Exact => "certified byte DFA",
        }
    }
}

fn vocabulary_width(vocabulary: &Vocabulary) -> Result<usize> {
    let maximum = vocabulary
        .tokens()
        .values()
        .flatten()
        .copied()
        .max()
        .map_or(vocabulary.eos_token_id(), |id| {
            id.max(vocabulary.eos_token_id())
        });
    usize::try_from(maximum)
        .ok()
        .and_then(|maximum| maximum.checked_add(1))
        .ok_or_else(|| {
            CompileError::ResourceLimitExceeded {
                stage: CompileStage::VocabularyProjection,
                observed: usize::MAX,
                limit: usize::MAX.saturating_sub(1),
            }
            .into()
        })
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index object with transitions:")?;
        for (state_id, token_ids) in self.transitions.iter() {
            writeln!(f, "{:?} -> {:#?}", state_id, token_ids)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{RegularExpression, DEAD_DFA_STATE};
    use crate::schema::{CompileLimits, SchemaPointer};

    fn compiled_dfa(pattern: &str) -> CompiledByteDfa {
        CompiledByteDfa::compile(
            &RegularExpression::Atom {
                forward: pattern.to_owned(),
                reverse: None,
            },
            &CompileLimits::default(),
            &SchemaPointer(String::new()),
        )
        .unwrap()
    }

    #[test]
    fn index_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let eos_token_id = 4;
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let initial_state = index.initial_state();
        assert_eq!(initial_state, 40);
        assert_eq!(index.final_states(), &HashSet::from_iter([24, 48, 56]));
        assert!(!index.is_final_state(&initial_state));

        let expected = HashMap::from_iter([
            (24, HashMap::from_iter([(3, 24), (4, 24), (2, 24)])),
            (48, HashMap::from_iter([(4, 48)])),
            (40, HashMap::from_iter([(3, 48), (2, 56)])),
            (56, HashMap::from_iter([(3, 24), (4, 56), (2, 24)])),
        ]);
        assert_eq!(index.transitions(), &expected);

        let allowed_tokens = index
            .allowed_tokens(&initial_state)
            .expect("No allowed tokens");
        let token_id = allowed_tokens.first().expect("No first tokens");

        let state = 48;
        assert_eq!(index.next_state(&initial_state, token_id), Some(state));
        assert!(index.is_final_state(&state));

        assert_eq!(index.next_state(&state, &eos_token_id), None);
        assert_eq!(index.next_state(&state, token_id), None);
    }

    #[test]
    fn index_from_regex_initital_in_allowed() {
        let regex = "`\\n(\\.\\n)?`\\n";
        let mut vocabulary = Vocabulary::new(104);
        for (token, token_id) in [("\n", 103), (".", 102), ("`", 101)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let allowed = index
            .allowed_tokens(&index.initial_state())
            .expect("No allowed tokens");
        assert!(allowed.contains(&101));
    }

    #[test]
    fn index_from_regex_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let mut vocabulary = Vocabulary::new(8);
        for (token, token_id) in [(" 😍", 5), ("blah", 0), ("😇", 2), ("😈a", 1), ("😍", 3)]
        {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        for (token, token_id) in [
            (vec![32, 240, 159, 152, 136], 7),
            (vec![32, 240, 159, 152, 141], 6),
            (vec![240, 159, 152, 141], 4),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.final_states(), &HashSet::from_iter([208, 128]));

        let expected = HashMap::from_iter([
            (
                208,
                HashMap::from_iter([(3, 208), (8, 208), (4, 208), (2, 208)]),
            ),
            (
                80,
                HashMap::from_iter([(2, 128), (7, 208), (5, 208), (6, 208)]),
            ),
            (128, HashMap::from_iter([(8, 128)])),
        ]);
        assert_eq!(index.transitions(), &expected);
    }

    #[test]
    fn index_incompatible_vocabulary_error() {
        let regex = "0 1";
        let mut vocabulary = Vocabulary::new(3);
        for (token, token_id) in [("0", 0), ("0 ", 1), ("1", 2)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let result = Index::new(regex, &vocabulary);
        assert!(result.is_err());

        if let Err(Error::IncompatibleVocabulary {
            regex: _,
            missing_tokens,
            ..
        }) = result
        {
            assert!(missing_tokens.contains(&" ".to_string()));
        } else {
            panic!("Expected IncompatibleVocabulary error");
        }
    }

    #[test]
    fn index_incompatible_vocabulary_error_non_ascii() {
        let regex = "😈😍";
        let mut vocabulary = Vocabulary::new(3);
        for (token, token_id) in [("😈", 0), (" ", 1), ("b", 2)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let result = Index::new(regex, &vocabulary);
        assert!(result.is_err());

        if let Err(Error::IncompatibleVocabulary {
            regex: _,
            missing_tokens,
            ..
        }) = result
        {
            assert!(missing_tokens.contains(&"\\xf0".to_string()));
        } else {
            panic!("Expected IncompatibleVocabulary error");
        }
    }

    #[test]
    fn index_from_regex_completeness() {
        let regex = "(ac|[^a])+";
        let eos_token_id = 3;
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (token, token_id) in [("a", 0), ("b", 1), ("c", 2)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let mut state = index.initial_state();

        // "acac" should be accepted
        for token_id in [0, 2, 0, 2] {
            state = index.next_state(&state, &token_id).expect("Transit failed");
        }
        assert!(index.is_final_state(&state));
    }

    #[test]
    fn certified_projection_supports_sparse_ids_and_byte_aliases() {
        let dfa = compiled_dfa(&regex::escape(r#""done""#));
        let mut vocabulary = Vocabulary::new(100);
        vocabulary.try_insert(r#""done""#, 42).unwrap();
        vocabulary.try_insert(r#""done""#, 96).unwrap();
        vocabulary.try_insert(r#"""#, 1).unwrap();
        vocabulary.try_insert("done", 2).unwrap();

        let index = Index::from_certified_dfa(&dfa, &vocabulary).unwrap();
        assert_eq!(index.initial_state(), 0);
        assert_eq!(index.vocab_size(), 101);
        let allowed = index.allowed_tokens(&0).unwrap();
        assert!(allowed.contains(&1));
        assert!(allowed.contains(&42));
        assert!(allowed.contains(&96));

        let final_state = index.next_state(&0, &42).unwrap();
        assert!(index.is_final_state(&final_state));
        assert!(index.allowed_tokens(&final_state).unwrap().contains(&100));
        assert_eq!(index.next_state(&final_state, &100), None);
    }

    #[test]
    fn legacy_projection_sizes_sparse_token_ids() {
        let mut vocabulary = Vocabulary::new(100);
        vocabulary.try_insert("x", 96).unwrap();
        let index = Index::new("x", &vocabulary).unwrap();
        assert_eq!(index.vocab_size(), 101);
        assert!(index
            .allowed_tokens(&index.initial_state())
            .unwrap()
            .contains(&96));
    }

    #[test]
    fn certified_projection_prunes_dead_token_prefixes() {
        let dfa = compiled_dfa("ab");
        let mut vocabulary = Vocabulary::new(3);
        vocabulary.try_insert("ax", 0).unwrap();
        vocabulary.try_insert("a", 1).unwrap();
        vocabulary.try_insert("b", 2).unwrap();

        let index = Index::from_certified_dfa(&dfa, &vocabulary).unwrap();
        let allowed = index.allowed_tokens(&0).unwrap();
        assert!(!allowed.contains(&0));
        assert!(allowed.contains(&1));
        let after_a = index.next_state(&0, &1).unwrap();
        assert_eq!(index.allowed_tokens(&after_a).unwrap(), vec![2]);
    }

    #[test]
    fn certified_projection_rejects_ambiguous_token_ids() {
        let dfa = compiled_dfa("a|bc");
        let mut vocabulary = Vocabulary::new(2);
        vocabulary.try_insert("a", 0).unwrap();
        vocabulary.try_insert("b", 0).unwrap();
        vocabulary.try_insert("c", 1).unwrap();

        assert!(matches!(
            Index::from_certified_dfa(&dfa, &vocabulary),
            Err(Error::AmbiguousTokenId { token_id: 0 })
        ));
    }

    #[test]
    fn empty_language_projects_to_an_empty_mask() {
        let dfa = CompiledByteDfa::compile(
            &RegularExpression::Empty,
            &CompileLimits::default(),
            &SchemaPointer(String::new()),
        )
        .unwrap();
        let mut vocabulary = Vocabulary::new(10);
        vocabulary.try_insert("anything", 9).unwrap();

        let index = Index::from_certified_dfa(&dfa, &vocabulary).unwrap();
        assert_eq!(index.initial_state(), 0);
        assert_eq!(index.vocab_size(), 11);
        assert!(index.final_states().is_empty());
        assert!(index.allowed_tokens(&0).is_none());
        assert_eq!(dfa.next_state(0, b'x'), DEAD_DFA_STATE);
    }
}
