//! Deterministic LALR(1) table construction and incremental recognition.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Serialize;

use crate::error::{CompileError, CompileStage, RuntimeError, RuntimeResource};
use crate::grammar::{
    CompiledTerminal, NonterminalId, ProductionId, ResidualGrammar, Symbol, TerminalCursorState,
    TerminalId,
};
use crate::schema::{CompileLimits, Provenance, RuntimeLimits};

pub type LalrStateId = u32;

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Lookahead {
    Terminal(TerminalId),
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lr1Item {
    pub production: ProductionId,
    pub dot: u32,
    pub lookahead: Lookahead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Action {
    Shift(LalrStateId),
    Reduce(ProductionId),
    Accept,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    ShiftReduce,
    ReduceReduce,
    AcceptConflict,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LalrConflict {
    pub kind: ConflictKind,
    pub state: LalrStateId,
    pub lookahead: Lookahead,
    pub first: Action,
    pub second: Action,
    pub productions: Vec<ProductionId>,
    pub provenance: Vec<Provenance>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LalrBuildStats {
    pub canonical_states: u64,
    pub merged_states: u64,
    pub lr_items: u64,
    pub action_entries: u64,
    pub goto_entries: u64,
    pub conflicts: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LalrRuntimeStats {
    pub shifts: u64,
    pub reductions: u64,
    pub max_stack_depth: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ParserProduction {
    lhs: NonterminalId,
    rhs: Vec<Symbol>,
    provenance: Provenance,
    original: Option<ProductionId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LalrTable {
    pub actions: Vec<Vec<(Lookahead, Action)>>,
    pub gotos: Vec<Vec<(NonterminalId, LalrStateId)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledLalr {
    grammar: Arc<ResidualGrammar>,
    productions: Vec<ParserProduction>,
    pub table: LalrTable,
    pub conflicts: Vec<LalrConflict>,
    pub lexical_safe: bool,
    pub stats: LalrBuildStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TerminalCandidate {
    terminal: TerminalId,
    cursor: TerminalCursorState,
    stack: Vec<LalrStateId>,
    shift: LalrStateId,
    reductions: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LalrCheckpoint {
    generation: u64,
    stack: Vec<LalrStateId>,
    candidates: Vec<TerminalCandidate>,
    byte_position: u32,
    stats: LalrRuntimeStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LalrRecognizer {
    compiled: Arc<CompiledLalr>,
    limits: RuntimeLimits,
    generation: u64,
    stack: Vec<LalrStateId>,
    candidates: Vec<TerminalCandidate>,
    byte_position: u32,
    stats: LalrRuntimeStats,
}

#[derive(Clone, Debug)]
struct TerminalSet {
    words: Vec<u64>,
    epsilon: bool,
}

impl TerminalSet {
    fn new(terminals: usize) -> Self {
        Self {
            words: vec![0; terminals.div_ceil(64)],
            epsilon: false,
        }
    }

    fn insert(&mut self, terminal: TerminalId) -> bool {
        let terminal = terminal as usize;
        let word = terminal / 64;
        let mask = 1_u64 << (terminal % 64);
        let changed = self.words[word] & mask == 0;
        self.words[word] |= mask;
        changed
    }

    fn union_without_epsilon(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            let next = *left | *right;
            changed |= next != *left;
            *left = next;
        }
        changed
    }

    fn terminals(&self) -> impl Iterator<Item = TerminalId> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, word)| {
                let word = *word;
                (0..64).filter_map(move |bit| {
                    (word & (1_u64 << bit) != 0)
                        .then(|| u32::try_from(word_index * 64 + bit).ok())
                        .flatten()
                })
            })
    }
}

impl CompiledLalr {
    pub fn build(
        grammar: Arc<ResidualGrammar>,
        limits: &CompileLimits,
    ) -> Result<Self, CompileError> {
        let augmented = u32::try_from(grammar.nonterminals.len()).map_err(|_| {
            CompileError::ResourceLimitExceeded {
                stage: CompileStage::LrConstruction,
                observed: grammar.nonterminals.len(),
                limit: u32::MAX as usize,
            }
        })?;
        let mut productions: Vec<_> = grammar
            .productions
            .iter()
            .map(|production| ParserProduction {
                lhs: production.lhs,
                rhs: production.rhs.clone(),
                provenance: production.provenance.clone(),
                original: Some(production.original),
            })
            .collect();
        let root_provenance = grammar
            .nonterminals
            .get(grammar.start as usize)
            .ok_or(CompileError::InternalInvariant {
                message: "residual start nonterminal does not exist",
            })?
            .provenance
            .clone();
        let augmented_production =
            u32::try_from(productions.len()).map_err(|_| CompileError::ResourceLimitExceeded {
                stage: CompileStage::LrConstruction,
                observed: productions.len(),
                limit: u32::MAX as usize,
            })?;
        productions.push(ParserProduction {
            lhs: augmented,
            rhs: vec![Symbol::Nonterminal(grammar.start)],
            provenance: root_provenance,
            original: None,
        });

        let mut by_lhs = grammar.productions_by_lhs.clone();
        by_lhs.push(vec![augmented_production]);
        let first = compute_first(&grammar)?;
        let initial = closure(
            vec![Lr1Item {
                production: augmented_production,
                dot: 0,
                lookahead: Lookahead::Eof,
            }],
            &productions,
            &by_lhs,
            &grammar.nullable,
            &first,
            limits,
        )?;

        let mut canonical = vec![initial.clone()];
        let mut state_ids = BTreeMap::from([(initial, 0_u32)]);
        let mut queue = VecDeque::from([0_u32]);
        let mut transitions = BTreeMap::new();
        let mut total_items = canonical[0].len();
        while let Some(state) = queue.pop_front() {
            let items = canonical[state as usize].clone();
            let mut symbols = BTreeSet::new();
            for item in &items {
                if let Some(symbol) = symbol_after_dot(&productions, *item)? {
                    symbols.insert(symbol_key(symbol));
                }
            }
            for key in symbols {
                let symbol = symbol_from_key(key);
                let target = goto(
                    &items,
                    symbol,
                    &productions,
                    &by_lhs,
                    &grammar.nullable,
                    &first,
                    limits,
                )?;
                let target_id = if let Some(id) = state_ids.get(&target) {
                    *id
                } else {
                    enforce_compile_limit(
                        CompileStage::LrConstruction,
                        canonical.len().saturating_add(1),
                        limits.max_lr_states,
                    )?;
                    total_items = total_items.checked_add(target.len()).ok_or(
                        CompileError::ResourceLimitExceeded {
                            stage: CompileStage::LrConstruction,
                            observed: usize::MAX,
                            limit: limits.max_lr_items,
                        },
                    )?;
                    enforce_compile_limit(
                        CompileStage::LrConstruction,
                        total_items,
                        limits.max_lr_items,
                    )?;
                    let id = u32::try_from(canonical.len()).map_err(|_| {
                        CompileError::ResourceLimitExceeded {
                            stage: CompileStage::LrConstruction,
                            observed: canonical.len(),
                            limit: u32::MAX as usize,
                        }
                    })?;
                    state_ids.insert(target.clone(), id);
                    canonical.push(target);
                    queue.push_back(id);
                    id
                };
                transitions.insert((state, key), target_id);
            }
        }

        let (merged, canonical_to_merged) = merge_cores(&canonical, limits)?;
        let mut merged_transitions = BTreeMap::new();
        for ((source, symbol), target) in transitions {
            let edge = (
                canonical_to_merged[source as usize],
                symbol,
                canonical_to_merged[target as usize],
            );
            if let Some(existing) = merged_transitions.insert((edge.0, edge.1), edge.2) {
                if existing != edge.2 {
                    return Err(CompileError::InternalInvariant {
                        message: "merged LR core has inconsistent transition targets",
                    });
                }
            }
        }
        let (table, conflicts) = build_table(
            &merged,
            &merged_transitions,
            &productions,
            augmented_production,
            limits,
        )?;
        let lexical_safe = lexical_safety(&grammar.terminals);
        let stats = LalrBuildStats {
            canonical_states: usize_to_u64(canonical.len()),
            merged_states: usize_to_u64(merged.len()),
            lr_items: usize_to_u64(total_items),
            action_entries: usize_to_u64(table.actions.iter().map(Vec::len).sum()),
            goto_entries: usize_to_u64(table.gotos.iter().map(Vec::len).sum()),
            conflicts: usize_to_u64(conflicts.len()),
        };
        Ok(Self {
            grammar,
            productions,
            table,
            conflicts,
            lexical_safe,
            stats,
        })
    }

    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.conflicts.is_empty() && self.lexical_safe
    }

    #[must_use]
    pub fn recognizer(self: &Arc<Self>, limits: RuntimeLimits) -> LalrRecognizer {
        LalrRecognizer {
            compiled: Arc::clone(self),
            limits,
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            stack: vec![0],
            candidates: Vec::new(),
            byte_position: 0,
            stats: LalrRuntimeStats {
                max_stack_depth: 1,
                ..LalrRuntimeStats::default()
            },
        }
    }
}

impl LalrRecognizer {
    pub fn advance_byte(&mut self, byte: u8) -> Result<bool, RuntimeError> {
        let next_position = self.byte_position as usize + 1;
        enforce_runtime_limit(
            RuntimeResource::InputBytes,
            next_position,
            self.limits.max_input_bytes,
        )?;
        if self.candidates.is_empty() {
            self.start_candidates()?;
        }
        let mut next = Vec::new();
        let mut accepted = Vec::new();
        for mut candidate in self.candidates.drain(..) {
            let terminal = &self.compiled.grammar.terminals[candidate.terminal as usize];
            let Some(cursor) = terminal.advance(candidate.cursor, byte) else {
                continue;
            };
            candidate.cursor = cursor;
            if terminal.is_accepting(cursor) {
                accepted.push(candidate.clone());
            }
            if terminal.is_live(cursor) {
                next.push(candidate);
            }
        }
        if accepted.len() > 1 {
            return Err(RuntimeError::InternalInvariant {
                message: "lexically safe LALR state accepted multiple terminals",
            });
        }
        if let Some(mut candidate) = accepted.pop() {
            candidate.stack.push(candidate.shift);
            enforce_runtime_limit(
                RuntimeResource::ParseStack,
                candidate.stack.len(),
                self.limits.max_parse_stack,
            )?;
            self.stack = candidate.stack;
            self.candidates.clear();
            self.stats.shifts = self.stats.shifts.saturating_add(1);
            self.stats.reductions = self.stats.reductions.saturating_add(candidate.reductions);
            self.stats.max_stack_depth = self
                .stats
                .max_stack_depth
                .max(usize_to_u64(self.stack.len()));
        } else {
            self.candidates = next;
            if self.candidates.is_empty() {
                return Ok(false);
            }
        }
        self.byte_position =
            u32::try_from(next_position).map_err(|_| RuntimeError::ResourceLimitExceeded {
                resource: RuntimeResource::InputBytes,
                observed: next_position,
                limit: u32::MAX as usize,
            })?;
        self.stats.bytes = self.stats.bytes.saturating_add(1);
        Ok(true)
    }

    fn start_candidates(&mut self) -> Result<(), RuntimeError> {
        let mut candidates = Vec::new();
        for terminal in 0..self.compiled.grammar.terminals.len() {
            let terminal_id =
                u32::try_from(terminal).map_err(|_| RuntimeError::InternalInvariant {
                    message: "terminal ID does not fit u32",
                })?;
            let mut stack = self.stack.clone();
            let reductions_before = self.stats.reductions;
            if let Some(Action::Shift(shift)) =
                self.action_after_reductions(&mut stack, Lookahead::Terminal(terminal_id))?
            {
                candidates.push(TerminalCandidate {
                    terminal: terminal_id,
                    cursor: self.compiled.grammar.terminals[terminal].start(),
                    stack,
                    shift,
                    reductions: self.stats.reductions.saturating_sub(reductions_before),
                });
            }
            self.stats.reductions = reductions_before;
        }
        enforce_runtime_limit(
            RuntimeResource::ActiveScans,
            candidates.len(),
            self.limits.max_active_scans,
        )?;
        self.candidates = candidates;
        Ok(())
    }

    fn action_after_reductions(
        &mut self,
        stack: &mut Vec<LalrStateId>,
        lookahead: Lookahead,
    ) -> Result<Option<Action>, RuntimeError> {
        loop {
            let state = *stack.last().ok_or(RuntimeError::InternalInvariant {
                message: "LALR stack is empty",
            })?;
            let action = find_action(&self.compiled.table, state, lookahead);
            match action {
                Some(Action::Reduce(production)) => {
                    let production = &self.compiled.productions[production as usize];
                    if production.rhs.len() >= stack.len() {
                        return Err(RuntimeError::InternalInvariant {
                            message: "LALR reduction underflowed the state stack",
                        });
                    }
                    stack.truncate(stack.len() - production.rhs.len());
                    let base = *stack.last().ok_or(RuntimeError::InternalInvariant {
                        message: "LALR stack is empty after reduction",
                    })?;
                    let target = find_goto(&self.compiled.table, base, production.lhs).ok_or(
                        RuntimeError::InternalInvariant {
                            message: "LALR goto is missing after reduction",
                        },
                    )?;
                    stack.push(target);
                    enforce_runtime_limit(
                        RuntimeResource::ParseStack,
                        stack.len(),
                        self.limits.max_parse_stack,
                    )?;
                    self.stats.reductions = self.stats.reductions.saturating_add(1);
                }
                other => return Ok(other),
            }
        }
    }

    #[must_use]
    pub fn is_accepting(&self) -> bool {
        if !self.candidates.is_empty() {
            return false;
        }
        let mut clone = self.clone();
        let mut stack = clone.stack.clone();
        matches!(
            clone.action_after_reductions(&mut stack, Lookahead::Eof),
            Ok(Some(Action::Accept))
        )
    }

    #[must_use]
    pub fn is_live(&self) -> bool {
        if !self.candidates.is_empty() {
            return true;
        }
        if self.is_accepting() {
            return true;
        }
        let mut clone = self.clone();
        clone.start_candidates().is_ok() && !clone.candidates.is_empty()
    }

    #[must_use]
    pub fn checkpoint(&self) -> LalrCheckpoint {
        LalrCheckpoint {
            generation: self.generation,
            stack: self.stack.clone(),
            candidates: self.candidates.clone(),
            byte_position: self.byte_position,
            stats: self.stats.clone(),
        }
    }

    pub fn restore(&mut self, checkpoint: &LalrCheckpoint) -> Result<(), RuntimeError> {
        if checkpoint.generation != self.generation {
            return Err(RuntimeError::InvalidCheckpoint {
                expected_generation: self.generation,
                found_generation: checkpoint.generation,
            });
        }
        self.stack.clone_from(&checkpoint.stack);
        self.candidates.clone_from(&checkpoint.candidates);
        self.byte_position = checkpoint.byte_position;
        self.stats.clone_from(&checkpoint.stats);
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> &LalrRuntimeStats {
        &self.stats
    }
}

fn compute_first(grammar: &ResidualGrammar) -> Result<Vec<TerminalSet>, CompileError> {
    let mut first: Vec<_> = (0..grammar.nonterminals.len())
        .map(|_| TerminalSet::new(grammar.terminals.len()))
        .collect();
    loop {
        let mut changed = false;
        for production in &grammar.productions {
            let lhs = production.lhs as usize;
            let mut prefix_nullable = true;
            for symbol in &production.rhs {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        changed |= first[lhs].insert(*terminal);
                        prefix_nullable = false;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        let rhs_first = first[*nonterminal as usize].clone();
                        changed |= first[lhs].union_without_epsilon(&rhs_first);
                        if !grammar.nullable[*nonterminal as usize] {
                            prefix_nullable = false;
                        }
                    }
                }
                if !prefix_nullable {
                    break;
                }
            }
            if prefix_nullable && !first[lhs].epsilon {
                first[lhs].epsilon = true;
                changed = true;
            }
        }
        if !changed {
            return Ok(first);
        }
    }
}

fn closure(
    seed: Vec<Lr1Item>,
    productions: &[ParserProduction],
    by_lhs: &[Vec<ProductionId>],
    nullable: &[bool],
    first: &[TerminalSet],
    limits: &CompileLimits,
) -> Result<Vec<Lr1Item>, CompileError> {
    let mut result: BTreeSet<_> = seed.into_iter().collect();
    let mut agenda: VecDeque<_> = result.iter().copied().collect();
    while let Some(item) = agenda.pop_front() {
        let Some(Symbol::Nonterminal(nonterminal)) = symbol_after_dot(productions, item)? else {
            continue;
        };
        let lookaheads = first_of_tail(item, productions, nullable, first)?;
        for production in &by_lhs[nonterminal as usize] {
            for lookahead in &lookaheads {
                let candidate = Lr1Item {
                    production: *production,
                    dot: 0,
                    lookahead: *lookahead,
                };
                if result.insert(candidate) {
                    enforce_compile_limit(
                        CompileStage::LrConstruction,
                        result.len(),
                        limits.max_lr_items,
                    )?;
                    agenda.push_back(candidate);
                }
            }
        }
    }
    Ok(result.into_iter().collect())
}

fn first_of_tail(
    item: Lr1Item,
    productions: &[ParserProduction],
    nullable: &[bool],
    first: &[TerminalSet],
) -> Result<Vec<Lookahead>, CompileError> {
    let production = &productions[item.production as usize];
    let start = item.dot as usize + 1;
    let mut result = BTreeSet::new();
    let mut tail_nullable = true;
    for symbol in production
        .rhs
        .get(start..)
        .ok_or(CompileError::InternalInvariant {
            message: "LR item dot is outside its production",
        })?
    {
        match symbol {
            Symbol::Terminal(terminal) => {
                result.insert(Lookahead::Terminal(*terminal));
                tail_nullable = false;
            }
            Symbol::Nonterminal(nonterminal) => {
                result.extend(
                    first[*nonterminal as usize]
                        .terminals()
                        .map(Lookahead::Terminal),
                );
                if !nullable[*nonterminal as usize] {
                    tail_nullable = false;
                }
            }
        }
        if !tail_nullable {
            break;
        }
    }
    if tail_nullable {
        result.insert(item.lookahead);
    }
    Ok(result.into_iter().collect())
}

fn goto(
    items: &[Lr1Item],
    symbol: Symbol,
    productions: &[ParserProduction],
    by_lhs: &[Vec<ProductionId>],
    nullable: &[bool],
    first: &[TerminalSet],
    limits: &CompileLimits,
) -> Result<Vec<Lr1Item>, CompileError> {
    let mut moved = Vec::new();
    for item in items {
        if symbol_after_dot(productions, *item)? == Some(symbol) {
            moved.push(Lr1Item {
                dot: item
                    .dot
                    .checked_add(1)
                    .ok_or(CompileError::InternalInvariant {
                        message: "LR item dot overflowed",
                    })?,
                ..*item
            });
        }
    }
    closure(moved, productions, by_lhs, nullable, first, limits)
}

fn symbol_after_dot(
    productions: &[ParserProduction],
    item: Lr1Item,
) -> Result<Option<Symbol>, CompileError> {
    let production =
        productions
            .get(item.production as usize)
            .ok_or(CompileError::InternalInvariant {
                message: "LR item production does not exist",
            })?;
    Ok(production.rhs.get(item.dot as usize).copied())
}

fn symbol_key(symbol: Symbol) -> (u8, u32) {
    match symbol {
        Symbol::Terminal(id) => (0, id),
        Symbol::Nonterminal(id) => (1, id),
    }
}

fn symbol_from_key(key: (u8, u32)) -> Symbol {
    if key.0 == 0 {
        Symbol::Terminal(key.1)
    } else {
        Symbol::Nonterminal(key.1)
    }
}

fn merge_cores(
    canonical: &[Vec<Lr1Item>],
    limits: &CompileLimits,
) -> Result<(Vec<Vec<Lr1Item>>, Vec<LalrStateId>), CompileError> {
    let mut groups: BTreeMap<Vec<(ProductionId, u32)>, LalrStateId> = BTreeMap::new();
    let mut merged_sets: Vec<BTreeSet<Lr1Item>> = Vec::new();
    let mut mapping = Vec::with_capacity(canonical.len());
    for state in canonical {
        let mut core: Vec<_> = state
            .iter()
            .map(|item| (item.production, item.dot))
            .collect();
        core.sort_unstable();
        core.dedup();
        let merged = if let Some(id) = groups.get(&core) {
            *id
        } else {
            enforce_compile_limit(
                CompileStage::LalrMerge,
                merged_sets.len().saturating_add(1),
                limits.max_lr_states,
            )?;
            let id = u32::try_from(merged_sets.len()).map_err(|_| {
                CompileError::ResourceLimitExceeded {
                    stage: CompileStage::LalrMerge,
                    observed: merged_sets.len(),
                    limit: u32::MAX as usize,
                }
            })?;
            groups.insert(core, id);
            merged_sets.push(BTreeSet::new());
            id
        };
        merged_sets[merged as usize].extend(state.iter().copied());
        mapping.push(merged);
    }
    Ok((
        merged_sets
            .into_iter()
            .map(|items| items.into_iter().collect())
            .collect(),
        mapping,
    ))
}

fn build_table(
    states: &[Vec<Lr1Item>],
    transitions: &BTreeMap<(LalrStateId, (u8, u32)), LalrStateId>,
    productions: &[ParserProduction],
    augmented: ProductionId,
    limits: &CompileLimits,
) -> Result<(LalrTable, Vec<LalrConflict>), CompileError> {
    let mut action_maps = vec![BTreeMap::new(); states.len()];
    let mut action_sources = vec![BTreeMap::new(); states.len()];
    let mut goto_maps = vec![BTreeMap::new(); states.len()];
    let mut conflicts = Vec::new();
    for (state_index, items) in states.iter().enumerate() {
        let state =
            u32::try_from(state_index).map_err(|_| CompileError::ResourceLimitExceeded {
                stage: CompileStage::LalrTable,
                observed: state_index,
                limit: u32::MAX as usize,
            })?;
        for item in items {
            match symbol_after_dot(productions, *item)? {
                Some(Symbol::Terminal(terminal)) => {
                    let target = transitions[&(state, symbol_key(Symbol::Terminal(terminal)))];
                    insert_action(
                        &mut action_maps[state_index],
                        state,
                        Lookahead::Terminal(terminal),
                        Action::Shift(target),
                        item.production,
                        productions,
                        &mut action_sources[state_index],
                        &mut conflicts,
                        limits,
                    )?;
                }
                Some(Symbol::Nonterminal(nonterminal)) => {
                    let target =
                        transitions[&(state, symbol_key(Symbol::Nonterminal(nonterminal)))];
                    goto_maps[state_index].insert(nonterminal, target);
                }
                None if item.production == augmented && item.lookahead == Lookahead::Eof => {
                    insert_action(
                        &mut action_maps[state_index],
                        state,
                        Lookahead::Eof,
                        Action::Accept,
                        item.production,
                        productions,
                        &mut action_sources[state_index],
                        &mut conflicts,
                        limits,
                    )?;
                }
                None => {
                    insert_action(
                        &mut action_maps[state_index],
                        state,
                        item.lookahead,
                        Action::Reduce(item.production),
                        item.production,
                        productions,
                        &mut action_sources[state_index],
                        &mut conflicts,
                        limits,
                    )?;
                }
            }
        }
    }
    let action_count: usize = action_maps.iter().map(BTreeMap::len).sum();
    let goto_count: usize = goto_maps.iter().map(BTreeMap::len).sum();
    enforce_compile_limit(
        CompileStage::LalrTable,
        action_count,
        limits.max_action_entries,
    )?;
    enforce_compile_limit(CompileStage::LalrTable, goto_count, limits.max_goto_entries)?;
    Ok((
        LalrTable {
            actions: action_maps
                .into_iter()
                .map(|row| row.into_iter().collect())
                .collect(),
            gotos: goto_maps
                .into_iter()
                .map(|row| row.into_iter().collect())
                .collect(),
        },
        conflicts,
    ))
}

#[allow(clippy::too_many_arguments)]
fn insert_action(
    row: &mut BTreeMap<Lookahead, Action>,
    state: LalrStateId,
    lookahead: Lookahead,
    action: Action,
    source: ProductionId,
    productions: &[ParserProduction],
    sources: &mut BTreeMap<Lookahead, Vec<ProductionId>>,
    conflicts: &mut Vec<LalrConflict>,
    limits: &CompileLimits,
) -> Result<(), CompileError> {
    let Some(existing) = row.get(&lookahead).copied() else {
        row.insert(lookahead, action);
        sources.insert(lookahead, vec![source]);
        return Ok(());
    };
    if existing == action {
        let entry = sources.entry(lookahead).or_default();
        if !entry.contains(&source) {
            entry.push(source);
        }
        return Ok(());
    }
    enforce_compile_limit(
        CompileStage::LalrTable,
        conflicts.len().saturating_add(1),
        limits.max_reported_conflicts,
    )?;
    let kind = match (existing, action) {
        (Action::Shift(_), Action::Reduce(_)) | (Action::Reduce(_), Action::Shift(_)) => {
            ConflictKind::ShiftReduce
        }
        (Action::Reduce(_), Action::Reduce(_)) => ConflictKind::ReduceReduce,
        _ => ConflictKind::AcceptConflict,
    };
    let mut involved = sources.get(&lookahead).cloned().unwrap_or_default();
    involved.push(source);
    for candidate in [existing, action] {
        if let Action::Reduce(production) = candidate {
            involved.push(production);
        }
    }
    involved.sort_unstable();
    involved.dedup();
    let provenance = involved
        .iter()
        .filter_map(|id| productions.get(*id as usize))
        .map(|production| production.provenance.clone())
        .collect();
    conflicts.push(LalrConflict {
        kind,
        state,
        lookahead,
        first: existing,
        second: action,
        productions: involved,
        provenance,
    });
    Ok(())
}

fn lexical_safety(terminals: &[CompiledTerminal]) -> bool {
    for terminal in terminals {
        if !terminal_is_prefix_free(terminal) {
            return false;
        }
    }
    for (index, left) in terminals.iter().enumerate() {
        for right in terminals.iter().skip(index + 1) {
            if terminal_languages_have_prefix_overlap(left, right) {
                return false;
            }
        }
    }
    true
}

fn terminal_is_prefix_free(terminal: &CompiledTerminal) -> bool {
    let CompiledTerminal::Dfa(dfa) = terminal else {
        return true;
    };
    for state in 0..dfa.state_count() {
        let Ok(state) = u32::try_from(state) else {
            return false;
        };
        if !dfa.is_accepting(state) {
            continue;
        }
        let mut queue = VecDeque::new();
        let mut seen = BTreeSet::new();
        for byte in 0_u8..=u8::MAX {
            let next = dfa.next_state(state, byte);
            if !dfa.is_dead(next) && seen.insert(next) {
                queue.push_back(next);
            }
        }
        while let Some(current) = queue.pop_front() {
            if dfa.is_accepting(current) {
                return false;
            }
            for byte in 0_u8..=u8::MAX {
                let next = dfa.next_state(current, byte);
                if !dfa.is_dead(next) && seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
    }
    true
}

fn terminal_languages_have_prefix_overlap(
    left: &CompiledTerminal,
    right: &CompiledTerminal,
) -> bool {
    let start = (left.start(), right.start());
    let mut queue = VecDeque::from([start]);
    let mut seen = BTreeSet::from([(cursor_key(start.0), cursor_key(start.1))]);
    while let Some((left_cursor, right_cursor)) = queue.pop_front() {
        for byte in 0_u8..=u8::MAX {
            let (Some(next_left), Some(next_right)) = (
                left.advance(left_cursor, byte),
                right.advance(right_cursor, byte),
            ) else {
                continue;
            };
            if left.is_accepting(next_left) || right.is_accepting(next_right) {
                return true;
            }
            let key = (cursor_key(next_left), cursor_key(next_right));
            if seen.insert(key) {
                queue.push_back((next_left, next_right));
            }
        }
    }
    false
}

fn cursor_key(cursor: TerminalCursorState) -> (u8, u32) {
    match cursor {
        TerminalCursorState::Literal(offset) => (0, offset),
        TerminalCursorState::Dfa(state) => (1, state),
    }
}

fn find_action(table: &LalrTable, state: LalrStateId, lookahead: Lookahead) -> Option<Action> {
    table
        .actions
        .get(state as usize)?
        .binary_search_by_key(&lookahead, |entry| entry.0)
        .ok()
        .map(|index| table.actions[state as usize][index].1)
}

fn find_goto(
    table: &LalrTable,
    state: LalrStateId,
    nonterminal: NonterminalId,
) -> Option<LalrStateId> {
    table
        .gotos
        .get(state as usize)?
        .binary_search_by_key(&nonterminal, |entry| entry.0)
        .ok()
        .map(|index| table.gotos[state as usize][index].1)
}

fn enforce_compile_limit(
    stage: CompileStage,
    observed: usize,
    limit: usize,
) -> Result<(), CompileError> {
    if observed > limit {
        Err(CompileError::ResourceLimitExceeded {
            stage,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}

fn enforce_runtime_limit(
    resource: RuntimeResource,
    observed: usize,
    limit: usize,
) -> Result<(), RuntimeError> {
    if observed > limit {
        Err(RuntimeError::ResourceLimitExceeded {
            resource,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::prepare_structural_for_test;
    use crate::grammar::{CompiledTerminal, ResidualNonterminal, ResidualProduction};
    use crate::schema::{CompileOptions, Provenance, ResourceId, SchemaPointer};

    fn grammar(
        nonterminal_count: usize,
        productions: Vec<(NonterminalId, Vec<Symbol>)>,
        literals: &[&[u8]],
    ) -> Arc<ResidualGrammar> {
        let provenance = Provenance {
            resource: ResourceId(0),
            pointer: SchemaPointer(String::new()),
            keyword: None,
        };
        let productions: Vec<_> = productions
            .into_iter()
            .enumerate()
            .map(|(id, (lhs, rhs))| ResidualProduction {
                id: u32::try_from(id).unwrap(),
                original: u32::try_from(id).unwrap(),
                lhs,
                rhs,
                provenance: provenance.clone(),
            })
            .collect();
        let mut productions_by_lhs = vec![Vec::new(); nonterminal_count];
        for production in &productions {
            productions_by_lhs[production.lhs as usize].push(production.id);
        }
        let productive_suffixes = productions
            .iter()
            .map(|production| {
                let mut suffixes = vec![false; production.rhs.len() + 1];
                suffixes[production.rhs.len()] = true;
                suffixes
            })
            .collect();
        Arc::new(ResidualGrammar {
            start: 0,
            nonterminals: (0..nonterminal_count)
                .map(|id| ResidualNonterminal {
                    id: u32::try_from(id).unwrap(),
                    original: u32::try_from(id).unwrap(),
                    name: format!("n{id}"),
                    provenance: provenance.clone(),
                })
                .collect(),
            terminals: literals
                .iter()
                .map(|bytes| CompiledTerminal::Literal(Arc::from(*bytes)))
                .collect(),
            terminal_provenance: literals.iter().map(|_| provenance.clone()).collect(),
            productions,
            productions_by_lhs,
            nullable: vec![false; nonterminal_count],
            productive_suffixes,
            collapsed_nonterminals: 0,
            collapsed_regions: 0,
            terminal_dfa_states: 0,
            terminal_dfa_bytes: 0,
        })
    }

    #[test]
    fn recursive_literal_array_builds_conflict_free_tables() {
        let schema = br##"{
          "$defs":{"node":{"anyOf":[{"const":null},{"type":"array","prefixItems":[{"$ref":"#/$defs/node"}],"items":false}]}},
          "$ref":"#/$defs/node"
        }"##;
        let residual = prepare_structural_for_test(schema, &CompileOptions::default()).unwrap();
        let compiled = CompiledLalr::build(Arc::new(residual), &CompileLimits::default()).unwrap();
        assert!(compiled.conflicts.is_empty());
        assert!(compiled.lexical_safe);
        let compiled = Arc::new(compiled);
        for value in ["null", "[null]", "[[null]]"] {
            let mut recognizer = compiled.recognizer(RuntimeLimits::default());
            for byte in value.bytes() {
                assert!(recognizer.advance_byte(byte).unwrap());
            }
            assert!(recognizer.is_accepting(), "{value}");
        }
    }

    #[test]
    fn reports_shift_reduce_conflicts_without_precedence() {
        let grammar = grammar(
            1,
            vec![
                (
                    0,
                    vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(0),
                    ],
                ),
                (0, vec![Symbol::Terminal(1)]),
            ],
            &[b"+", b"n"],
        );
        let compiled = CompiledLalr::build(grammar, &CompileLimits::default()).unwrap();

        assert!(!compiled.is_usable());
        assert!(compiled
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::ShiftReduce));
        assert!(compiled
            .conflicts
            .iter()
            .all(|conflict| !conflict.provenance.is_empty()));
    }

    #[test]
    fn reports_reduce_reduce_conflicts_with_both_productions() {
        let grammar = grammar(
            3,
            vec![
                (0, vec![Symbol::Nonterminal(1)]),
                (0, vec![Symbol::Nonterminal(2)]),
                (1, vec![Symbol::Terminal(0)]),
                (2, vec![Symbol::Terminal(0)]),
            ],
            &[b"x"],
        );
        let compiled = CompiledLalr::build(grammar, &CompileLimits::default()).unwrap();
        let conflict = compiled
            .conflicts
            .iter()
            .find(|conflict| conflict.kind == ConflictKind::ReduceReduce)
            .unwrap();

        assert_eq!(conflict.productions.len(), 2);
        assert_eq!(conflict.provenance.len(), 2);
    }

    #[test]
    fn table_and_conflict_limits_return_typed_errors() {
        let deterministic = grammar(
            2,
            vec![
                (0, vec![Symbol::Nonterminal(1)]),
                (1, vec![Symbol::Terminal(0)]),
            ],
            &[b"x"],
        );
        for (limits, expected_stage) in [
            (
                CompileLimits {
                    max_lr_items: 0,
                    ..CompileLimits::default()
                },
                CompileStage::LrConstruction,
            ),
            (
                CompileLimits {
                    max_action_entries: 0,
                    ..CompileLimits::default()
                },
                CompileStage::LalrTable,
            ),
            (
                CompileLimits {
                    max_goto_entries: 0,
                    ..CompileLimits::default()
                },
                CompileStage::LalrTable,
            ),
        ] {
            assert!(matches!(
                CompiledLalr::build(Arc::clone(&deterministic), &limits),
                Err(CompileError::ResourceLimitExceeded { stage, .. })
                    if stage == expected_stage
            ));
        }

        let ambiguous = grammar(
            3,
            vec![
                (0, vec![Symbol::Nonterminal(1)]),
                (0, vec![Symbol::Nonterminal(2)]),
                (1, vec![Symbol::Terminal(0)]),
                (2, vec![Symbol::Terminal(0)]),
            ],
            &[b"x"],
        );
        let limits = CompileLimits {
            max_reported_conflicts: 0,
            ..CompileLimits::default()
        };
        assert!(matches!(
            CompiledLalr::build(ambiguous, &limits),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::LalrTable,
                ..
            })
        ));
    }

    #[test]
    fn runtime_stack_and_scan_limits_return_typed_errors() {
        let grammar = grammar(1, vec![(0, vec![Symbol::Terminal(0)])], &[b"x"]);
        let compiled =
            Arc::new(CompiledLalr::build(Arc::clone(&grammar), &CompileLimits::default()).unwrap());
        let mut recognizer = compiled.recognizer(RuntimeLimits {
            max_parse_stack: 1,
            ..RuntimeLimits::default()
        });
        assert!(matches!(
            recognizer.advance_byte(b'x'),
            Err(RuntimeError::ResourceLimitExceeded {
                resource: RuntimeResource::ParseStack,
                ..
            })
        ));

        let mut recognizer = compiled.recognizer(RuntimeLimits {
            max_active_scans: 0,
            ..RuntimeLimits::default()
        });
        assert!(matches!(
            recognizer.advance_byte(b'x'),
            Err(RuntimeError::ResourceLimitExceeded {
                resource: RuntimeResource::ActiveScans,
                ..
            })
        ));
    }
}
