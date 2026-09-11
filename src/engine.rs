//! Schema compiler orchestration and backend selection.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::earley::{CompiledEarley, EarleyCheckpoint, EarleyRecognizer, EarleyStats};
use crate::error::CompileError;
use crate::grammar::{
    self, CertificateFailureReason, CompiledByteDfa, Grammar, NonterminalId,
    RegularCertificateKind, SccId,
};
use crate::index::Index;
use crate::lalr::LalrConflict;
use crate::lalr::{CompiledLalr, LalrCheckpoint, LalrRecognizer, LalrRuntimeStats};
use crate::schema::{
    CompileOptions, DiagnosticLocation, RuntimeLimits, SchemaArena, SchemaPointer,
    CANONICAL_POLICY_ID, COMPILED_FORMAT_VERSION, DRAFT_2020_12, PROFILE_ID,
};
use crate::vocabulary::reverse::DerivedVocabulary;
use crate::vocabulary::Vocabulary;
use crate::Result;

static NEXT_DFA_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    WholeDfa,
    Lalr,
    Earley,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendPolicy {
    #[default]
    Auto,
    ForceDfa,
    ForceLalr,
    ForceEarley,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub location: DiagnosticLocation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SccReport {
    pub id: SccId,
    pub members: Vec<NonterminalId>,
    pub source_pointers: Vec<SchemaPointer>,
    pub certificate: Option<RegularCertificateKind>,
    pub failure_reason: Option<CertificateFailureReason>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TierReport {
    pub format_version: u32,
    pub dialect: String,
    pub profile: String,
    pub canonical_policy: String,
    pub schema_nodes: usize,
    pub reference_edges: usize,
    pub grammar_symbols: usize,
    pub grammar_productions: usize,
    pub sccs: Vec<SccReport>,
    pub selected_backend: BackendKind,
    pub nfa_states: Option<usize>,
    pub nfa_transitions: Option<usize>,
    pub dfa_states: Option<usize>,
    pub dfa_bytes: Option<usize>,
    pub collapsed_regular_regions: usize,
    pub compiled_terminals: usize,
    pub terminal_dfa_states: usize,
    pub terminal_dfa_bytes: usize,
    pub canonical_lr_states: usize,
    pub lalr_states: usize,
    pub action_entries: usize,
    pub goto_entries: usize,
    pub conflicts: usize,
    pub conflict_summaries: Vec<LalrConflict>,
    pub nullable_nonterminals: usize,
    pub leo_eligible_rules: usize,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CompileProfile {
    pub schema_parse_ns: u64,
    pub normalization_ns: u64,
    pub grammar_lowering_ns: u64,
    pub grammar_reduction_ns: u64,
    pub scc_analysis_ns: u64,
    pub regular_certification_ns: u64,
    pub nfa_construction_ns: u64,
    pub dfa_determinization_ns: u64,
    pub vocabulary_projection_ns: u64,
    pub residual_grammar_ns: u64,
    pub lalr_construction_ns: u64,
    pub earley_preparation_ns: u64,
    pub vocabulary_trie_ns: u64,
    pub total_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SemanticContract {
    pub format_version: u32,
    pub dialect: String,
    pub profile: String,
    pub canonical_policy: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompiledBackend {
    Dfa(Arc<Index>),
    Lalr(Arc<CompiledLalr>),
    Earley(Arc<CompiledEarley>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledSchema {
    pub semantic_contract: SemanticContract,
    pub normalized: SchemaArena,
    pub grammar: Grammar,
    pub report: TierReport,
    pub backend: CompiledBackend,
    byte_dfa: Option<Arc<CompiledByteDfa>>,
    vocabulary: Option<Arc<DerivedVocabulary>>,
}

enum PreparedBackend {
    Dfa(CompiledByteDfa),
    Lalr(Arc<CompiledLalr>),
    Earley(Arc<CompiledEarley>),
}

struct PreparedCompilation {
    arena: SchemaArena,
    grammar: Grammar,
    report: TierReport,
    backend: PreparedBackend,
    profile: CompileProfile,
}

impl CompiledSchema {
    pub fn analyze(schema: &[u8], options: &CompileOptions) -> Result<TierReport, CompileError> {
        Ok(prepare(schema, options, BackendPolicy::Auto)?.report)
    }

    pub fn compile(
        schema: &[u8],
        vocabulary: &Vocabulary,
        options: &CompileOptions,
    ) -> Result<Self> {
        Self::compile_profiled(schema, vocabulary, options).map(|(compiled, _)| compiled)
    }

    pub fn compile_profiled(
        schema: &[u8],
        vocabulary: &Vocabulary,
        options: &CompileOptions,
    ) -> Result<(Self, CompileProfile)> {
        Self::compile_profiled_with_policy(schema, vocabulary, options, BackendPolicy::Auto)
    }

    pub fn compile_with_policy(
        schema: &[u8],
        vocabulary: &Vocabulary,
        options: &CompileOptions,
        policy: BackendPolicy,
    ) -> Result<Self> {
        Self::compile_profiled_with_policy(schema, vocabulary, options, policy)
            .map(|(compiled, _)| compiled)
    }

    fn compile_profiled_with_policy(
        schema: &[u8],
        vocabulary: &Vocabulary,
        options: &CompileOptions,
        policy: BackendPolicy,
    ) -> Result<(Self, CompileProfile)> {
        let total_started = Instant::now();
        let mut prepared = prepare(schema, options, policy)?;
        let (backend, byte_dfa, derived_vocabulary, vocabulary_projection_ns) = match prepared
            .backend
        {
            PreparedBackend::Dfa(dfa) => {
                let projection_started = Instant::now();
                let index = Index::from_certified_dfa(&dfa, vocabulary)?;
                (
                    CompiledBackend::Dfa(Arc::new(index)),
                    Some(Arc::new(dfa)),
                    None,
                    elapsed_ns(projection_started),
                )
            }
            PreparedBackend::Lalr(lalr) => {
                let trie_started = Instant::now();
                let vocabulary = Arc::new(DerivedVocabulary::build(vocabulary, &options.limits)?);
                prepared.profile.vocabulary_trie_ns = elapsed_ns(trie_started);
                (CompiledBackend::Lalr(lalr), None, Some(vocabulary), 0)
            }
            PreparedBackend::Earley(earley) => {
                let trie_started = Instant::now();
                let vocabulary = Arc::new(DerivedVocabulary::build(vocabulary, &options.limits)?);
                prepared.profile.vocabulary_trie_ns = elapsed_ns(trie_started);
                (CompiledBackend::Earley(earley), None, Some(vocabulary), 0)
            }
        };
        prepared.profile.vocabulary_projection_ns = vocabulary_projection_ns;
        prepared.profile.total_ns = elapsed_ns(total_started);
        Ok((
            Self {
                semantic_contract: semantic_contract(),
                normalized: prepared.arena,
                grammar: prepared.grammar,
                report: prepared.report,
                backend,
                byte_dfa,
                vocabulary: derived_vocabulary,
            },
            prepared.profile,
        ))
    }

    pub fn index(&self) -> Result<&Index, CompileError> {
        match &self.backend {
            CompiledBackend::Dfa(index) => Ok(index),
            CompiledBackend::Lalr(_) | CompiledBackend::Earley(_) => {
                let location = self.grammar.nonterminals[self.grammar.start as usize]
                    .provenance
                    .location();
                Err(CompileError::StructuralBackendRequired { location })
            }
        }
    }

    pub fn recognizer(
        &self,
        limits: RuntimeLimits,
    ) -> Result<RecognizerState, crate::error::RuntimeError> {
        match &self.backend {
            CompiledBackend::Dfa(_) => Ok(RecognizerState::Dfa(DfaRecognizer {
                dfa: Arc::clone(self.byte_dfa.as_ref().ok_or(
                    crate::error::RuntimeError::InternalInvariant {
                        message: "whole-DFA backend has no byte automaton",
                    },
                )?),
                state: 0,
                position: 0,
                generation: NEXT_DFA_GENERATION.fetch_add(1, Ordering::Relaxed),
                limits,
            })),
            CompiledBackend::Lalr(compiled) => {
                Ok(RecognizerState::Lalr(compiled.recognizer(limits)))
            }
            CompiledBackend::Earley(compiled) => {
                Ok(RecognizerState::Earley(compiled.recognizer(limits, true)?))
            }
        }
    }

    pub fn guide(
        &self,
        max_rollback: usize,
        limits: RuntimeLimits,
    ) -> Result<Guide, crate::error::RuntimeError> {
        Guide::from_compiled(self, max_rollback, limits)
    }
}

fn prepare(
    schema: &[u8],
    options: &CompileOptions,
    policy: BackendPolicy,
) -> Result<PreparedCompilation, CompileError> {
    let parse_started = Instant::now();
    let root = crate::schema::parse_schema(schema, options)?;
    let schema_parse_ns = elapsed_ns(parse_started);

    let normalization_started = Instant::now();
    let arena = crate::schema::normalize_schema(root, options)?;
    let normalization_ns = elapsed_ns(normalization_started);

    let lowering_started = Instant::now();
    let mut grammar = grammar::lower(&arena, &options.limits)?;
    let grammar_lowering_ns = elapsed_ns(lowering_started);

    let reduction_started = Instant::now();
    grammar::reduce(&mut grammar)?;
    let grammar_reduction_ns = elapsed_ns(reduction_started);

    let scc_started = Instant::now();
    let sccs = grammar::analyze_sccs(&grammar)?;
    let scc_analysis_ns = elapsed_ns(scc_started);

    let certification_started = Instant::now();
    let analysis = grammar::certify_regular_with_sccs(&grammar, sccs)?;
    let regular_certification_ns = elapsed_ns(certification_started);

    let root = grammar.nonterminals.get(grammar.start as usize).ok_or(
        CompileError::InternalInvariant {
            message: "reduced grammar start does not exist",
        },
    )?;
    let root_location = root.provenance.location();
    let (dfa, automaton_profile) = match analysis.whole_language.as_ref() {
        Some(expression) => {
            let (dfa, profile) = CompiledByteDfa::compile_profiled(
                expression,
                &options.limits,
                &root.provenance.pointer,
            )?;
            (Some(dfa), profile)
        }
        None => (None, grammar::AutomatonTimings::default()),
    };
    if policy == BackendPolicy::ForceDfa && dfa.is_none() {
        return Err(CompileError::BackendUnavailable {
            backend: "whole_dfa",
            location: root_location,
            reason: "the whole grammar has no exact regular certificate".to_owned(),
        });
    }
    let mut residual_grammar_ns = 0;
    let mut lalr_construction_ns = 0;
    let mut earley_preparation_ns = 0;
    let use_whole_dfa =
        dfa.is_some() && matches!(policy, BackendPolicy::Auto | BackendPolicy::ForceDfa);
    let (backend, structural_metrics) = if use_whole_dfa {
        let dfa = dfa.ok_or(CompileError::InternalInvariant {
            message: "whole-DFA policy lost its compiled automaton",
        })?;
        (PreparedBackend::Dfa(dfa), None)
    } else {
        let residual_started = Instant::now();
        let residual = Arc::new(grammar::build_residual(
            &grammar,
            &analysis,
            &options.limits,
        )?);
        residual_grammar_ns = elapsed_ns(residual_started);
        if policy == BackendPolicy::ForceEarley {
            let started = Instant::now();
            let earley = Arc::new(CompiledEarley::build(
                Arc::clone(&residual),
                &options.limits,
            )?);
            earley_preparation_ns = elapsed_ns(started);
            let metrics = StructuralMetrics::from_earley(&residual, &earley, None);
            (PreparedBackend::Earley(earley), Some(metrics))
        } else {
            let lalr_started = Instant::now();
            let lalr = CompiledLalr::build(Arc::clone(&residual), &options.limits);
            lalr_construction_ns = elapsed_ns(lalr_started);
            match (policy, lalr) {
                (BackendPolicy::ForceLalr, Ok(lalr)) if !lalr.is_usable() => {
                    return Err(CompileError::BackendUnavailable {
                        backend: "lalr",
                        location: root_location,
                        reason: format!(
                            "{} parser conflicts; lexical safety={}",
                            lalr.conflicts.len(),
                            lalr.lexical_safe
                        ),
                    });
                }
                (BackendPolicy::ForceLalr, Err(error)) => return Err(error),
                (BackendPolicy::ForceLalr, Ok(lalr)) => {
                    let metrics = StructuralMetrics::from_lalr(&residual, &lalr);
                    (PreparedBackend::Lalr(Arc::new(lalr)), Some(metrics))
                }
                (_, Ok(lalr)) if lalr.is_usable() => {
                    let metrics = StructuralMetrics::from_lalr(&residual, &lalr);
                    (PreparedBackend::Lalr(Arc::new(lalr)), Some(metrics))
                }
                (_, lalr_result) => {
                    let lalr_metrics = lalr_result.as_ref().ok();
                    let started = Instant::now();
                    let earley = Arc::new(CompiledEarley::build(
                        Arc::clone(&residual),
                        &options.limits,
                    )?);
                    earley_preparation_ns = elapsed_ns(started);
                    let metrics = StructuralMetrics::from_earley(&residual, &earley, lalr_metrics);
                    (PreparedBackend::Earley(earley), Some(metrics))
                }
            }
        }
    };
    let report = build_report(
        &arena,
        &grammar,
        &analysis,
        &backend,
        structural_metrics.as_ref(),
    )?;

    Ok(PreparedCompilation {
        arena,
        grammar,
        report,
        backend,
        profile: CompileProfile {
            schema_parse_ns,
            normalization_ns,
            grammar_lowering_ns,
            grammar_reduction_ns,
            scc_analysis_ns,
            regular_certification_ns,
            nfa_construction_ns: duration_ns(automaton_profile.nfa),
            dfa_determinization_ns: duration_ns(automaton_profile.dfa),
            vocabulary_projection_ns: 0,
            residual_grammar_ns,
            lalr_construction_ns,
            earley_preparation_ns,
            vocabulary_trie_ns: 0,
            total_ns: 0,
        },
    })
}

fn elapsed_ns(started: Instant) -> u64 {
    duration_ns(started.elapsed())
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn build_report(
    arena: &SchemaArena,
    grammar: &Grammar,
    analysis: &grammar::RegularAnalysis,
    backend: &PreparedBackend,
    structural: Option<&StructuralMetrics>,
) -> Result<TierReport, CompileError> {
    let grammar_symbols = grammar
        .nonterminals
        .len()
        .checked_add(grammar.terminals.len())
        .ok_or(CompileError::InternalInvariant {
            message: "grammar symbol count overflowed",
        })?;
    let mut sccs = Vec::new();
    let mut diagnostics = Vec::new();
    sccs.try_reserve_exact(analysis.sccs.components.len())
        .map_err(|_| CompileError::ResourceLimitExceeded {
            stage: crate::error::CompileStage::RegularCertification,
            observed: analysis.sccs.components.len(),
            limit: analysis.sccs.components.len().saturating_sub(1),
        })?;

    for component in &analysis.sccs.components {
        let certificate = analysis.certificates.get(component.id as usize).ok_or(
            CompileError::InternalInvariant {
                message: "SCC certificate does not exist",
            },
        )?;
        let mut source_pointers = Vec::new();
        source_pointers
            .try_reserve_exact(component.members.len())
            .map_err(|_| CompileError::ResourceLimitExceeded {
                stage: crate::error::CompileStage::RegularCertification,
                observed: component.members.len(),
                limit: component.members.len().saturating_sub(1),
            })?;
        for member in &component.members {
            source_pointers.push(
                grammar
                    .nonterminals
                    .get(*member as usize)
                    .ok_or(CompileError::InternalInvariant {
                        message: "SCC member does not exist",
                    })?
                    .provenance
                    .pointer
                    .clone(),
            );
        }
        source_pointers.sort();
        source_pointers.dedup();
        if let Some(reason) = certificate.failure_reason {
            let pointer = source_pointers
                .first()
                .cloned()
                .unwrap_or_else(|| SchemaPointer(String::new()));
            diagnostics.push(Diagnostic {
                code: "regular_certificate_failed".to_owned(),
                message: certificate_failure_message(reason).to_owned(),
                location: DiagnosticLocation {
                    resource: crate::schema::ResourceId(0),
                    pointer,
                    keyword: None,
                },
            });
        }
        sccs.push(SccReport {
            id: component.id,
            members: component.members.clone(),
            source_pointers,
            certificate: certificate.kind,
            failure_reason: certificate.failure_reason,
        });
    }
    diagnostics.sort_by(|left, right| {
        left.location
            .pointer
            .cmp(&right.location.pointer)
            .then_with(|| left.code.cmp(&right.code))
    });

    Ok(TierReport {
        format_version: COMPILED_FORMAT_VERSION,
        dialect: DRAFT_2020_12.to_owned(),
        profile: PROFILE_ID.to_owned(),
        canonical_policy: CANONICAL_POLICY_ID.to_owned(),
        schema_nodes: arena.nodes.len(),
        reference_edges: arena.reference_edges.len(),
        grammar_symbols,
        grammar_productions: grammar.productions.len(),
        sccs,
        selected_backend: match backend {
            PreparedBackend::Dfa(_) => BackendKind::WholeDfa,
            PreparedBackend::Lalr(_) => BackendKind::Lalr,
            PreparedBackend::Earley(_) => BackendKind::Earley,
        },
        nfa_states: match backend {
            PreparedBackend::Dfa(dfa) => Some(dfa.nfa_state_count()),
            _ => None,
        },
        nfa_transitions: match backend {
            PreparedBackend::Dfa(dfa) => Some(dfa.nfa_transition_count()),
            _ => None,
        },
        dfa_states: match backend {
            PreparedBackend::Dfa(dfa) => Some(dfa.state_count()),
            _ => None,
        },
        dfa_bytes: match backend {
            PreparedBackend::Dfa(dfa) => Some(dfa.memory_bytes()),
            _ => None,
        },
        collapsed_regular_regions: structural
            .map_or(0, |metrics| metrics.collapsed_regular_regions),
        compiled_terminals: structural.map_or(0, |metrics| metrics.compiled_terminals),
        terminal_dfa_states: structural.map_or(0, |metrics| metrics.terminal_dfa_states),
        terminal_dfa_bytes: structural.map_or(0, |metrics| metrics.terminal_dfa_bytes),
        canonical_lr_states: structural.map_or(0, |metrics| metrics.canonical_lr_states),
        lalr_states: structural.map_or(0, |metrics| metrics.lalr_states),
        action_entries: structural.map_or(0, |metrics| metrics.action_entries),
        goto_entries: structural.map_or(0, |metrics| metrics.goto_entries),
        conflicts: structural.map_or(0, |metrics| metrics.conflicts),
        conflict_summaries: structural
            .map_or_else(Vec::new, |metrics| metrics.conflict_summaries.clone()),
        nullable_nonterminals: structural.map_or(0, |metrics| metrics.nullable_nonterminals),
        leo_eligible_rules: structural.map_or(0, |metrics| metrics.leo_eligible_rules),
        diagnostics,
    })
}

#[derive(Clone, Debug)]
struct StructuralMetrics {
    collapsed_regular_regions: usize,
    compiled_terminals: usize,
    terminal_dfa_states: usize,
    terminal_dfa_bytes: usize,
    canonical_lr_states: usize,
    lalr_states: usize,
    action_entries: usize,
    goto_entries: usize,
    conflicts: usize,
    conflict_summaries: Vec<LalrConflict>,
    nullable_nonterminals: usize,
    leo_eligible_rules: usize,
}

impl StructuralMetrics {
    fn from_lalr(residual: &grammar::ResidualGrammar, lalr: &CompiledLalr) -> Self {
        Self {
            collapsed_regular_regions: residual.collapsed_regions,
            compiled_terminals: residual.terminals.len(),
            terminal_dfa_states: residual.terminal_dfa_states,
            terminal_dfa_bytes: residual.terminal_dfa_bytes,
            canonical_lr_states: lalr.stats.canonical_states as usize,
            lalr_states: lalr.stats.merged_states as usize,
            action_entries: lalr.stats.action_entries as usize,
            goto_entries: lalr.stats.goto_entries as usize,
            conflicts: lalr.conflicts.len(),
            conflict_summaries: lalr.conflicts.clone(),
            nullable_nonterminals: residual.nullable.iter().filter(|value| **value).count(),
            leo_eligible_rules: 0,
        }
    }

    fn from_earley(
        residual: &grammar::ResidualGrammar,
        earley: &CompiledEarley,
        lalr: Option<&CompiledLalr>,
    ) -> Self {
        let mut metrics = lalr.map_or_else(
            || Self {
                collapsed_regular_regions: residual.collapsed_regions,
                compiled_terminals: residual.terminals.len(),
                terminal_dfa_states: residual.terminal_dfa_states,
                terminal_dfa_bytes: residual.terminal_dfa_bytes,
                canonical_lr_states: 0,
                lalr_states: 0,
                action_entries: 0,
                goto_entries: 0,
                conflicts: 0,
                conflict_summaries: Vec::new(),
                nullable_nonterminals: earley.nullable_nonterminals,
                leo_eligible_rules: earley.leo_eligible_rules,
            },
            |lalr| Self::from_lalr(residual, lalr),
        );
        metrics.nullable_nonterminals = earley.nullable_nonterminals;
        metrics.leo_eligible_rules = earley.leo_eligible_rules;
        metrics
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advance {
    Rejected,
    Live,
    Accepting,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum RecognizerStats {
    Dfa { bytes: u64 },
    Lalr(LalrRuntimeStats),
    Earley(EarleyStats),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfaCheckpoint {
    generation: u64,
    state: u32,
    position: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecognizerCheckpoint {
    Dfa(DfaCheckpoint),
    Lalr(LalrCheckpoint),
    Earley(EarleyCheckpoint),
}

pub trait Recognizer {
    type Checkpoint: Clone;

    fn try_advance_bytes(&mut self, bytes: &[u8]) -> Result<Advance, crate::error::RuntimeError>;
    fn is_accepting(&self) -> bool;
    fn is_live(&self) -> bool;
    fn checkpoint(&self) -> Self::Checkpoint;
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<(), crate::error::RuntimeError>;
    fn stats(&self) -> RecognizerStats;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfaRecognizer {
    dfa: Arc<CompiledByteDfa>,
    state: u32,
    position: u32,
    generation: u64,
    limits: RuntimeLimits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecognizerState {
    Dfa(DfaRecognizer),
    Lalr(LalrRecognizer),
    Earley(EarleyRecognizer),
}

impl RecognizerState {
    pub fn try_advance_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<Advance, crate::error::RuntimeError> {
        let checkpoint = self.checkpoint();
        for byte in bytes {
            let result = match self {
                Self::Dfa(recognizer) => recognizer.advance_byte(*byte),
                Self::Lalr(recognizer) => recognizer.advance_byte(*byte),
                Self::Earley(recognizer) => recognizer.advance_byte(*byte),
            };
            let live = match result {
                Ok(live) => live,
                Err(error) => {
                    self.restore(&checkpoint)?;
                    return Err(error);
                }
            };
            if !live {
                self.restore(&checkpoint)?;
                return Ok(Advance::Rejected);
            }
        }
        Ok(if self.is_accepting() {
            Advance::Accepting
        } else {
            Advance::Live
        })
    }

    #[must_use]
    pub fn is_accepting(&self) -> bool {
        match self {
            Self::Dfa(recognizer) => recognizer.dfa.is_accepting(recognizer.state),
            Self::Lalr(recognizer) => recognizer.is_accepting(),
            Self::Earley(recognizer) => recognizer.is_accepting(),
        }
    }

    #[must_use]
    pub fn is_live(&self) -> bool {
        match self {
            Self::Dfa(recognizer) => recognizer.dfa.is_live(recognizer.state),
            Self::Lalr(recognizer) => recognizer.is_live(),
            Self::Earley(recognizer) => recognizer.is_live(),
        }
    }

    #[must_use]
    pub fn checkpoint(&self) -> RecognizerCheckpoint {
        match self {
            Self::Dfa(recognizer) => RecognizerCheckpoint::Dfa(DfaCheckpoint {
                generation: recognizer.generation,
                state: recognizer.state,
                position: recognizer.position,
            }),
            Self::Lalr(recognizer) => RecognizerCheckpoint::Lalr(recognizer.checkpoint()),
            Self::Earley(recognizer) => RecognizerCheckpoint::Earley(recognizer.checkpoint()),
        }
    }

    pub fn restore(
        &mut self,
        checkpoint: &RecognizerCheckpoint,
    ) -> Result<(), crate::error::RuntimeError> {
        match (self, checkpoint) {
            (Self::Dfa(recognizer), RecognizerCheckpoint::Dfa(checkpoint)) => {
                if checkpoint.generation != recognizer.generation {
                    return Err(crate::error::RuntimeError::InvalidCheckpoint {
                        expected_generation: recognizer.generation,
                        found_generation: checkpoint.generation,
                    });
                }
                recognizer.state = checkpoint.state;
                recognizer.position = checkpoint.position;
                Ok(())
            }
            (Self::Lalr(recognizer), RecognizerCheckpoint::Lalr(checkpoint)) => {
                recognizer.restore(checkpoint)
            }
            (Self::Earley(recognizer), RecognizerCheckpoint::Earley(checkpoint)) => {
                recognizer.restore(checkpoint)
            }
            _ => Err(crate::error::RuntimeError::InternalInvariant {
                message: "checkpoint belongs to a different recognizer backend",
            }),
        }
    }

    #[must_use]
    pub fn stats(&self) -> RecognizerStats {
        match self {
            Self::Dfa(recognizer) => RecognizerStats::Dfa {
                bytes: recognizer.position as u64,
            },
            Self::Lalr(recognizer) => RecognizerStats::Lalr(recognizer.stats().clone()),
            Self::Earley(recognizer) => RecognizerStats::Earley(recognizer.stats().clone()),
        }
    }
}

impl Recognizer for RecognizerState {
    type Checkpoint = RecognizerCheckpoint;

    fn try_advance_bytes(&mut self, bytes: &[u8]) -> Result<Advance, crate::error::RuntimeError> {
        RecognizerState::try_advance_bytes(self, bytes)
    }

    fn is_accepting(&self) -> bool {
        RecognizerState::is_accepting(self)
    }

    fn is_live(&self) -> bool {
        RecognizerState::is_live(self)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        RecognizerState::checkpoint(self)
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<(), crate::error::RuntimeError> {
        RecognizerState::restore(self, checkpoint)
    }

    fn stats(&self) -> RecognizerStats {
        RecognizerState::stats(self)
    }
}

impl DfaRecognizer {
    fn advance_byte(&mut self, byte: u8) -> Result<bool, crate::error::RuntimeError> {
        let next_position = self.position as usize + 1;
        if next_position > self.limits.max_input_bytes {
            return Err(crate::error::RuntimeError::ResourceLimitExceeded {
                resource: crate::error::RuntimeResource::InputBytes,
                observed: next_position,
                limit: self.limits.max_input_bytes,
            });
        }
        let next = self.dfa.next_state(self.state, byte);
        if self.dfa.is_dead(next) || !self.dfa.is_live(next) {
            return Ok(false);
        }
        self.state = next;
        self.position = u32::try_from(next_position).map_err(|_| {
            crate::error::RuntimeError::ResourceLimitExceeded {
                resource: crate::error::RuntimeResource::InputBytes,
                observed: next_position,
                limit: u32::MAX as usize,
            }
        })?;
        Ok(true)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MaskStats {
    pub queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: u64,
    pub trie_edges_visited: u64,
    pub trie_edges_pruned: u64,
    pub token_ids_enabled: u64,
    pub peak_stack_depth: u64,
    pub errors: u64,
}

#[derive(Clone, Debug)]
enum GuideRuntime {
    Dfa { index: Arc<Index>, state: u32 },
    Structural { recognizer: Box<RecognizerState> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GuideCheckpoint {
    Dfa {
        state: u32,
        committed_tokens: usize,
        committed_bytes: usize,
        finished: bool,
    },
    Structural {
        recognizer: RecognizerCheckpoint,
        committed_tokens: usize,
        committed_bytes: usize,
        finished: bool,
    },
}

#[derive(Clone, Debug)]
pub struct Guide {
    runtime: GuideRuntime,
    vocabulary: Option<Arc<DerivedVocabulary>>,
    eos_token_id: u32,
    vocab_size: usize,
    mask_words: usize,
    initial: GuideCheckpoint,
    history: VecDeque<GuideCheckpoint>,
    max_rollback: usize,
    finished: bool,
    committed_token_ids: Vec<u32>,
    committed_bytes: Vec<u8>,
    limits: RuntimeLimits,
    mask_stats: MaskStats,
    cached_mask: Option<Vec<u32>>,
}

impl Guide {
    pub fn from_index(
        index: Arc<Index>,
        max_rollback: usize,
    ) -> std::result::Result<Self, crate::error::RuntimeError> {
        let limits = RuntimeLimits::default();
        validate_rollback_limit(max_rollback, &limits)?;
        let state = index.initial_state();
        let vocab_size = index.vocab_size();
        let eos_token_id = index.eos_token_id();
        let initial = GuideCheckpoint::Dfa {
            state,
            committed_tokens: 0,
            committed_bytes: 0,
            finished: false,
        };
        let mut history = VecDeque::new();
        history.try_reserve_exact(max_rollback).map_err(|_| {
            crate::error::RuntimeError::AllocationFailed {
                resource: crate::error::RuntimeResource::CheckpointHistory,
            }
        })?;
        Ok(Self {
            runtime: GuideRuntime::Dfa { index, state },
            vocabulary: None,
            eos_token_id,
            vocab_size,
            mask_words: vocab_size.div_ceil(32),
            initial,
            history,
            max_rollback,
            finished: false,
            committed_token_ids: Vec::new(),
            committed_bytes: Vec::new(),
            limits,
            mask_stats: MaskStats::default(),
            cached_mask: None,
        })
    }

    fn from_compiled(
        compiled: &CompiledSchema,
        max_rollback: usize,
        limits: RuntimeLimits,
    ) -> std::result::Result<Self, crate::error::RuntimeError> {
        validate_rollback_limit(max_rollback, &limits)?;
        let (runtime, vocabulary, eos_token_id, vocab_size, mask_words) = match &compiled.backend {
            CompiledBackend::Dfa(index) => (
                GuideRuntime::Dfa {
                    index: Arc::clone(index),
                    state: index.initial_state(),
                },
                None,
                index.eos_token_id(),
                index.vocab_size(),
                index.vocab_size().div_ceil(32),
            ),
            CompiledBackend::Lalr(_) | CompiledBackend::Earley(_) => {
                let vocabulary = Arc::clone(compiled.vocabulary.as_ref().ok_or(
                    crate::error::RuntimeError::InternalInvariant {
                        message: "structural backend has no derived vocabulary",
                    },
                )?);
                (
                    GuideRuntime::Structural {
                        recognizer: Box::new(compiled.recognizer(limits.clone())?),
                    },
                    Some(Arc::clone(&vocabulary)),
                    vocabulary.eos_token_id,
                    vocabulary.vocab_size,
                    vocabulary.mask_words,
                )
            }
        };
        let initial = match &runtime {
            GuideRuntime::Dfa { state, .. } => GuideCheckpoint::Dfa {
                state: *state,
                committed_tokens: 0,
                committed_bytes: 0,
                finished: false,
            },
            GuideRuntime::Structural { recognizer } => GuideCheckpoint::Structural {
                recognizer: recognizer.checkpoint(),
                committed_tokens: 0,
                committed_bytes: 0,
                finished: false,
            },
        };
        let mut history = VecDeque::new();
        history.try_reserve_exact(max_rollback).map_err(|_| {
            crate::error::RuntimeError::AllocationFailed {
                resource: crate::error::RuntimeResource::CheckpointHistory,
            }
        })?;
        Ok(Self {
            runtime,
            vocabulary,
            eos_token_id,
            vocab_size,
            mask_words,
            initial,
            history,
            max_rollback,
            finished: false,
            committed_token_ids: Vec::new(),
            committed_bytes: Vec::new(),
            limits,
            mask_stats: MaskStats::default(),
            cached_mask: None,
        })
    }

    #[must_use]
    pub fn backend(&self) -> BackendKind {
        match &self.runtime {
            GuideRuntime::Dfa { .. } => BackendKind::WholeDfa,
            GuideRuntime::Structural { recognizer } => match recognizer.as_ref() {
                RecognizerState::Dfa(_) => BackendKind::WholeDfa,
                RecognizerState::Lalr(_) => BackendKind::Lalr,
                RecognizerState::Earley(_) => BackendKind::Earley,
            },
        }
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[must_use]
    pub fn mask_words(&self) -> usize {
        self.mask_words
    }

    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    #[must_use]
    pub fn position(&self) -> usize {
        self.committed_bytes.len()
    }

    #[must_use]
    pub fn state_fingerprint(&self) -> u64 {
        let mut hash = if self.finished {
            0xcbf2_9ce4_8422_2324 ^ 1
        } else {
            0xcbf2_9ce4_8422_2324
        };
        for token_id in &self.committed_token_ids {
            for byte in token_id.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
        }
        for byte in &self.committed_bytes {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    pub fn dfa_state(&self) -> std::result::Result<u32, crate::error::RuntimeError> {
        match &self.runtime {
            GuideRuntime::Dfa { state, .. } => Ok(*state),
            GuideRuntime::Structural { .. } => Err(crate::error::RuntimeError::StateIdUnavailable),
        }
    }

    #[must_use]
    pub fn is_accepting(&self) -> bool {
        if self.finished {
            return false;
        }
        match &self.runtime {
            GuideRuntime::Dfa { index, state } => index.is_final_state(state),
            GuideRuntime::Structural { recognizer } => recognizer.is_accepting(),
        }
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    #[must_use]
    pub fn get_allowed_rollback(&self) -> usize {
        self.history.len()
    }

    #[must_use]
    pub fn committed_token_ids(&self) -> &[u32] {
        &self.committed_token_ids
    }

    #[must_use]
    pub fn committed_bytes(&self) -> &[u8] {
        &self.committed_bytes
    }

    #[must_use]
    pub fn parser_stats(&self) -> Option<RecognizerStats> {
        match &self.runtime {
            GuideRuntime::Dfa { .. } => None,
            GuideRuntime::Structural { recognizer } => Some(recognizer.stats()),
        }
    }

    #[must_use]
    pub fn mask_stats(&self) -> &MaskStats {
        &self.mask_stats
    }

    pub fn fill_mask(
        &mut self,
        destination: &mut [u32],
    ) -> std::result::Result<(), crate::error::RuntimeError> {
        if destination.len() < self.mask_words {
            return Err(crate::error::RuntimeError::InternalInvariant {
                message: "mask destination is shorter than mask_words",
            });
        }
        destination.fill(0);
        self.mask_stats.queries = self.mask_stats.queries.saturating_add(1);
        if let Some(cached) = &self.cached_mask {
            destination[..self.mask_words].copy_from_slice(cached);
            self.mask_stats.cache_hits = self.mask_stats.cache_hits.saturating_add(1);
            return Ok(());
        }
        self.mask_stats.cache_misses = self.mask_stats.cache_misses.saturating_add(1);
        if self.finished {
            return self.remember_mask(destination);
        }
        let result = match &mut self.runtime {
            GuideRuntime::Dfa { index, state } => {
                if let Some(tokens) = index.allowed_tokens_iter(state) {
                    for &token in tokens {
                        set_mask_bit(destination, token)?;
                        self.mask_stats.token_ids_enabled =
                            self.mask_stats.token_ids_enabled.saturating_add(1);
                    }
                }
                Ok(())
            }
            GuideRuntime::Structural { recognizer } => {
                let vocabulary = self.vocabulary.as_ref().ok_or(
                    crate::error::RuntimeError::InternalInvariant {
                        message: "structural guide has no derived vocabulary",
                    },
                )?;
                let result = fill_structural_mask(
                    recognizer,
                    vocabulary,
                    destination,
                    &self.limits,
                    &mut self.mask_stats,
                );
                if result.is_err() {
                    destination.fill(0);
                    self.mask_stats.errors = self.mask_stats.errors.saturating_add(1);
                }
                result
            }
        };
        if result.is_ok() {
            self.remember_mask(destination)?;
        }
        result
    }

    pub fn get_tokens(&mut self) -> std::result::Result<Vec<u32>, crate::error::RuntimeError> {
        let mut mask = vec![0u32; self.mask_words];
        self.fill_mask(&mut mask)?;
        let mut tokens = Vec::new();
        for token in 0..self.vocab_size {
            if mask[token / 32] & (1u32 << (token % 32)) != 0 {
                tokens.push(u32::try_from(token).map_err(|_| {
                    crate::error::RuntimeError::InternalInvariant {
                        message: "vocabulary width exceeds TokenId",
                    }
                })?);
            }
        }
        Ok(tokens)
    }

    #[must_use]
    pub fn accepts_tokens(&self, sequence: &[u32]) -> bool {
        if sequence.contains(&self.eos_token_id) {
            return false;
        }
        let mut trial = self.clone();
        sequence.iter().all(|token| trial.advance(*token).is_ok())
    }

    pub fn advance(
        &mut self,
        token_id: u32,
    ) -> std::result::Result<(), crate::error::RuntimeError> {
        if self.finished {
            return Err(crate::error::RuntimeError::GuideFinished);
        }
        let next_count = self.committed_token_ids.len().checked_add(1).ok_or(
            crate::error::RuntimeError::ResourceLimitExceeded {
                resource: crate::error::RuntimeResource::CommittedTokens,
                observed: usize::MAX,
                limit: self.limits.max_committed_tokens,
            },
        )?;
        if next_count > self.limits.max_committed_tokens {
            return Err(crate::error::RuntimeError::ResourceLimitExceeded {
                resource: crate::error::RuntimeResource::CommittedTokens,
                observed: next_count,
                limit: self.limits.max_committed_tokens,
            });
        }
        self.committed_token_ids.try_reserve(1).map_err(|_| {
            crate::error::RuntimeError::AllocationFailed {
                resource: crate::error::RuntimeResource::CommittedTokens,
            }
        })?;
        let before = self.checkpoint();
        if token_id == self.eos_token_id {
            if !self.is_accepting() {
                return Err(crate::error::RuntimeError::EosNotAccepting);
            }
            self.retain_checkpoint(before);
            self.finished = true;
            self.committed_token_ids.push(token_id);
            self.invalidate_mask();
            return Ok(());
        }

        let token_bytes = self
            .vocabulary
            .as_ref()
            .and_then(|vocabulary| vocabulary.shared_bytes(token_id));
        if matches!(self.runtime, GuideRuntime::Structural { .. }) && token_bytes.is_none() {
            return Err(crate::error::RuntimeError::UnknownTokenId { token_id });
        }
        if let Some(bytes) = &token_bytes {
            self.committed_bytes.try_reserve(bytes.len()).map_err(|_| {
                crate::error::RuntimeError::AllocationFailed {
                    resource: crate::error::RuntimeResource::InputBytes,
                }
            })?;
        }
        match &mut self.runtime {
            GuideRuntime::Dfa { index, state } => {
                let next = index
                    .next_state(state, &token_id)
                    .ok_or(crate::error::RuntimeError::TokenNotAllowed { token_id })?;
                *state = next;
            }
            GuideRuntime::Structural { recognizer } => {
                let bytes = token_bytes.as_deref().ok_or(
                    crate::error::RuntimeError::InternalInvariant {
                        message: "validated structural token bytes disappeared",
                    },
                )?;
                match recognizer.try_advance_bytes(bytes)? {
                    Advance::Rejected => {
                        return Err(crate::error::RuntimeError::TokenNotAllowed { token_id });
                    }
                    Advance::Live | Advance::Accepting => {}
                }
            }
        }
        self.retain_checkpoint(before);
        if let Some(bytes) = token_bytes {
            self.committed_bytes.extend_from_slice(&bytes);
        }
        self.committed_token_ids.push(token_id);
        self.invalidate_mask();
        Ok(())
    }

    pub fn rollback(
        &mut self,
        count: usize,
    ) -> std::result::Result<(), crate::error::RuntimeError> {
        if count == 0 {
            return Ok(());
        }
        if count > self.history.len() {
            return Err(crate::error::RuntimeError::RollbackUnavailable {
                requested: count,
                available: self.history.len(),
            });
        }
        let mut target = None;
        for _ in 0..count {
            target = self.history.pop_back();
        }
        self.restore_guide(target.as_ref().ok_or(
            crate::error::RuntimeError::InternalInvariant {
                message: "rollback history changed during rollback",
            },
        )?)?;
        self.invalidate_mask();
        Ok(())
    }

    pub fn reset(&mut self) -> std::result::Result<(), crate::error::RuntimeError> {
        let initial = self.initial.clone();
        self.restore_guide(&initial)?;
        self.history.clear();
        self.mask_stats = MaskStats::default();
        self.cached_mask = None;
        Ok(())
    }

    fn remember_mask(
        &mut self,
        destination: &[u32],
    ) -> std::result::Result<(), crate::error::RuntimeError> {
        let mut cached = Vec::new();
        cached.try_reserve_exact(self.mask_words).map_err(|_| {
            crate::error::RuntimeError::AllocationFailed {
                resource: crate::error::RuntimeResource::MaskTraversal,
            }
        })?;
        cached.extend_from_slice(&destination[..self.mask_words]);
        self.mask_stats.cache_bytes = u64::try_from(self.mask_words)
            .unwrap_or(u64::MAX)
            .saturating_mul(4);
        self.cached_mask = Some(cached);
        Ok(())
    }

    fn invalidate_mask(&mut self) {
        self.cached_mask = None;
        self.mask_stats.cache_bytes = 0;
    }

    fn checkpoint(&self) -> GuideCheckpoint {
        match &self.runtime {
            GuideRuntime::Dfa { state, .. } => GuideCheckpoint::Dfa {
                state: *state,
                committed_tokens: self.committed_token_ids.len(),
                committed_bytes: self.committed_bytes.len(),
                finished: self.finished,
            },
            GuideRuntime::Structural { recognizer } => GuideCheckpoint::Structural {
                recognizer: recognizer.checkpoint(),
                committed_tokens: self.committed_token_ids.len(),
                committed_bytes: self.committed_bytes.len(),
                finished: self.finished,
            },
        }
    }

    fn retain_checkpoint(&mut self, checkpoint: GuideCheckpoint) {
        if self.max_rollback == 0 {
            return;
        }
        while self.history.len() >= self.max_rollback {
            self.history.pop_front();
        }
        self.history.push_back(checkpoint);
    }

    fn restore_guide(
        &mut self,
        checkpoint: &GuideCheckpoint,
    ) -> std::result::Result<(), crate::error::RuntimeError> {
        let (committed_tokens, committed_bytes, finished) = match (&mut self.runtime, checkpoint) {
            (
                GuideRuntime::Dfa { state, .. },
                GuideCheckpoint::Dfa {
                    state: saved,
                    committed_tokens,
                    committed_bytes,
                    finished,
                },
            ) => {
                *state = *saved;
                (*committed_tokens, *committed_bytes, *finished)
            }
            (
                GuideRuntime::Structural { recognizer },
                GuideCheckpoint::Structural {
                    recognizer: saved,
                    committed_tokens,
                    committed_bytes,
                    finished,
                },
            ) => {
                recognizer.restore(saved)?;
                (*committed_tokens, *committed_bytes, *finished)
            }
            _ => {
                return Err(crate::error::RuntimeError::InternalInvariant {
                    message: "guide checkpoint belongs to a different backend",
                });
            }
        };
        self.committed_token_ids.truncate(committed_tokens);
        self.committed_bytes.truncate(committed_bytes);
        self.finished = finished;
        Ok(())
    }
}

#[derive(Clone)]
struct TrieFrame {
    node: usize,
    next_edge: usize,
    emitted: bool,
    checkpoint: RecognizerCheckpoint,
}

fn fill_structural_mask(
    recognizer: &mut RecognizerState,
    vocabulary: &DerivedVocabulary,
    destination: &mut [u32],
    limits: &RuntimeLimits,
    stats: &mut MaskStats,
) -> std::result::Result<(), crate::error::RuntimeError> {
    let root = recognizer.checkpoint();
    let result = (|| {
        if recognizer.is_accepting() {
            set_mask_bit(destination, vocabulary.eos_token_id)?;
            stats.token_ids_enabled = stats.token_ids_enabled.saturating_add(1);
        }
        let mut stack = vec![TrieFrame {
            node: 0,
            next_edge: 0,
            emitted: false,
            checkpoint: root.clone(),
        }];
        let mut visited = 0usize;
        while !stack.is_empty() {
            let depth = u64::try_from(stack.len()).unwrap_or(u64::MAX);
            stats.peak_stack_depth = stats.peak_stack_depth.max(depth);
            let frame_index = stack.len() - 1;
            let node_index = stack[frame_index].node;
            let node = vocabulary.trie.nodes.get(node_index).ok_or(
                crate::error::RuntimeError::InternalInvariant {
                    message: "token trie node is out of bounds",
                },
            )?;
            if !stack[frame_index].emitted {
                if recognizer.is_live() {
                    let start = usize::try_from(node.token_start).map_err(|_| {
                        crate::error::RuntimeError::InternalInvariant {
                            message: "token trie terminal offset does not fit usize",
                        }
                    })?;
                    let token_len = usize::try_from(node.token_len).map_err(|_| {
                        crate::error::RuntimeError::InternalInvariant {
                            message: "token trie terminal length does not fit usize",
                        }
                    })?;
                    let end = start.checked_add(token_len).ok_or(
                        crate::error::RuntimeError::InternalInvariant {
                            message: "token trie terminal range overflowed",
                        },
                    )?;
                    for &token_id in vocabulary.trie.terminal_ids.get(start..end).ok_or(
                        crate::error::RuntimeError::InternalInvariant {
                            message: "token trie terminal range is invalid",
                        },
                    )? {
                        set_mask_bit(destination, token_id)?;
                        stats.token_ids_enabled = stats.token_ids_enabled.saturating_add(1);
                    }
                }
                stack[frame_index].emitted = true;
            }

            let edge_len = usize::try_from(node.edge_len).map_err(|_| {
                crate::error::RuntimeError::InternalInvariant {
                    message: "token trie edge length does not fit usize",
                }
            })?;
            if stack[frame_index].next_edge >= edge_len {
                let checkpoint = stack[frame_index].checkpoint.clone();
                recognizer.restore(&checkpoint)?;
                stack.pop();
                continue;
            }

            let edge_offset = stack[frame_index].next_edge;
            stack[frame_index].next_edge += 1;
            let edge_start = usize::try_from(node.edge_start).map_err(|_| {
                crate::error::RuntimeError::InternalInvariant {
                    message: "token trie edge offset does not fit usize",
                }
            })?;
            let edge_index = edge_start.checked_add(edge_offset).ok_or(
                crate::error::RuntimeError::InternalInvariant {
                    message: "token trie edge range overflowed",
                },
            )?;
            let edge = *vocabulary.trie.edges.get(edge_index).ok_or(
                crate::error::RuntimeError::InternalInvariant {
                    message: "token trie edge is out of bounds",
                },
            )?;
            let observed = visited.saturating_add(1);
            if observed > limits.max_mask_trie_edges {
                return Err(crate::error::RuntimeError::ResourceLimitExceeded {
                    resource: crate::error::RuntimeResource::MaskTraversal,
                    observed,
                    limit: limits.max_mask_trie_edges,
                });
            }
            visited = observed;
            stats.trie_edges_visited = stats.trie_edges_visited.saturating_add(1);
            let checkpoint = stack[frame_index].checkpoint.clone();
            recognizer.restore(&checkpoint)?;
            if recognizer.try_advance_bytes(&[edge.byte])? == Advance::Rejected {
                stats.trie_edges_pruned = stats.trie_edges_pruned.saturating_add(1);
                continue;
            }
            stack.push(TrieFrame {
                node: usize::try_from(edge.child).map_err(|_| {
                    crate::error::RuntimeError::InternalInvariant {
                        message: "token trie child does not fit usize",
                    }
                })?,
                next_edge: 0,
                emitted: false,
                checkpoint: recognizer.checkpoint(),
            });
        }
        Ok(())
    })();
    let restored = recognizer.restore(&root);
    match (result, restored) {
        (_, Err(error)) => Err(error),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn set_mask_bit(
    destination: &mut [u32],
    token_id: u32,
) -> std::result::Result<(), crate::error::RuntimeError> {
    let token =
        usize::try_from(token_id).map_err(|_| crate::error::RuntimeError::InternalInvariant {
            message: "token ID does not fit usize",
        })?;
    let word = token / 32;
    let bit = token % 32;
    let slot = destination
        .get_mut(word)
        .ok_or(crate::error::RuntimeError::InternalInvariant {
            message: "token ID exceeds the validated mask width",
        })?;
    *slot |= 1u32 << bit;
    Ok(())
}

fn validate_rollback_limit(
    max_rollback: usize,
    limits: &RuntimeLimits,
) -> std::result::Result<(), crate::error::RuntimeError> {
    if max_rollback > limits.max_checkpoint_history {
        Err(crate::error::RuntimeError::ResourceLimitExceeded {
            resource: crate::error::RuntimeResource::CheckpointHistory,
            observed: max_rollback,
            limit: limits.max_checkpoint_history,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn prepare_structural_for_test(
    schema: &[u8],
    options: &CompileOptions,
) -> Result<grammar::ResidualGrammar, CompileError> {
    let root = crate::schema::parse_schema(schema, options)?;
    let arena = crate::schema::normalize_schema(root, options)?;
    let mut grammar = grammar::lower(&arena, &options.limits)?;
    grammar::reduce(&mut grammar)?;
    let analysis = grammar::certify_regular(&grammar)?;
    grammar::build_residual(&grammar, &analysis, &options.limits)
}

fn semantic_contract() -> SemanticContract {
    SemanticContract {
        format_version: COMPILED_FORMAT_VERSION,
        dialect: DRAFT_2020_12.to_owned(),
        profile: PROFILE_ID.to_owned(),
        canonical_policy: CANONICAL_POLICY_ID.to_owned(),
    }
}

fn certificate_failure_message(reason: CertificateFailureReason) -> &'static str {
    match reason {
        CertificateFailureReason::MultipleRecursiveOccurrences => {
            "a production has multiple recursive occurrences"
        }
        CertificateFailureReason::MixedLinearOrientation => {
            "the component mixes left-linear and right-linear recursion"
        }
        CertificateFailureReason::RecursiveSymbolInInterior => {
            "a recursive symbol occurs inside a production"
        }
        CertificateFailureReason::UncertifiedDependency => {
            "a dependency has no exact regular certificate"
        }
        CertificateFailureReason::UnsupportedRegularOperation => {
            "an exact regular operation is unavailable"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary(tokens: &[(&str, u32)], eos: u32) -> Vocabulary {
        let mut vocabulary = Vocabulary::new(eos);
        for (bytes, token) in tokens {
            vocabulary.try_insert(*bytes, *token).unwrap();
        }
        vocabulary
    }

    fn accepts(compiled: &CompiledSchema, bytes: &[u8]) -> bool {
        let mut recognizer = compiled.recognizer(RuntimeLimits::default()).unwrap();
        !matches!(
            recognizer.try_advance_bytes(bytes).unwrap(),
            Advance::Rejected
        ) && recognizer.is_accepting()
    }

    #[test]
    fn deep_acyclic_references_compile_to_the_whole_dfa() {
        let schema = include_bytes!("../testdata/regressions/deep_acyclic_ref.json");
        let vocabulary = vocabulary(&[(r#""done""#, 0), (r#""other""#, 1)], 2);
        let compiled =
            CompiledSchema::compile(schema, &vocabulary, &CompileOptions::default()).unwrap();

        assert_eq!(compiled.report.reference_edges, 5);
        assert_eq!(compiled.report.selected_backend, BackendKind::WholeDfa);
        let index = compiled.index().unwrap();
        let final_state = index.next_state(&index.initial_state(), &0).unwrap();
        assert!(index.is_final_state(&final_state));
        assert_eq!(index.next_state(&index.initial_state(), &1), None);
    }

    #[test]
    fn recursive_schema_selects_a_structural_backend() {
        let schema = include_bytes!("../testdata/regressions/recursive_optional_property.json");
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[("null", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();

        assert!(matches!(
            compiled.report.selected_backend,
            BackendKind::Lalr | BackendKind::Earley
        ));
        assert!(compiled.normalized.nodes.iter().any(|node| {
            node.provenance.pointer.0.ends_with("/properties/next")
                && matches!(node.kind, crate::schema::NormalizedSchema::Ref(_))
        }));
        assert!(!compiled.report.diagnostics.is_empty());
        assert!(matches!(
            compiled.index(),
            Err(CompileError::StructuralBackendRequired { .. })
        ));
    }

    #[test]
    fn reports_are_byte_identical() {
        let schema = include_bytes!("../testdata/regressions/recursive_required_property.json");
        let first = CompiledSchema::analyze(schema, &CompileOptions::default()).unwrap();
        let second = CompiledSchema::analyze(schema, &CompileOptions::default()).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn all_of_compiles_as_intersection() {
        let schema = include_bytes!("../testdata/regressions/allof_intersection.json");
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[(r#""x""#, 0), (r#""y""#, 1)], 2),
            &CompileOptions::default(),
        )
        .unwrap();
        let index = compiled.index().unwrap();

        let accepted = index.next_state(&index.initial_state(), &0).unwrap();
        assert!(index.is_final_state(&accepted));
        assert_eq!(index.next_state(&index.initial_state(), &1), None);
    }

    #[test]
    fn prefix_items_compile_to_the_exact_language() {
        let schema = include_bytes!("../testdata/regressions/prefix_items.json");
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(
                &[
                    ("[1]", 0),
                    ("[1,2]", 1),
                    ("[1,2,2]", 2),
                    ("[]", 3),
                    ("[2]", 4),
                    ("[1,3]", 5),
                    ("[1,2,2,2]", 6),
                ],
                7,
            ),
            &CompileOptions::default(),
        )
        .unwrap();
        let index = compiled.index().unwrap();
        let allowed = index.allowed_tokens(&index.initial_state()).unwrap();

        assert!(allowed.contains(&0));
        assert!(allowed.contains(&1));
        assert!(allowed.contains(&2));
        assert!(!allowed.iter().any(|token| (3..=6).contains(token)));
        for token in 0..=2 {
            let state = index.next_state(&index.initial_state(), &token).unwrap();
            assert!(index.is_final_state(&state));
        }
    }

    #[test]
    fn false_schema_compiles_to_an_empty_index() {
        let compiled = CompiledSchema::compile(
            b"false",
            &vocabulary(&[("null", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();

        assert_eq!(compiled.report.selected_backend, BackendKind::WholeDfa);
        assert!(compiled
            .index()
            .unwrap()
            .allowed_tokens(&compiled.index().unwrap().initial_state())
            .is_none());
    }

    #[test]
    fn open_object_language_is_never_weakened() {
        let error = CompiledSchema::analyze(
            br#"{"type":"object","properties":{"x":{"const":1}}}"#,
            &CompileOptions::default(),
        )
        .unwrap_err();

        assert!(matches!(error, CompileError::UnsupportedCombination { .. }));
        assert!(error.to_string().contains("canonical key-order memory"));
    }

    #[test]
    fn profiled_compilation_accounts_for_every_stage() {
        let vocabulary = vocabulary(&[(r#""done""#, 0)], 1);
        let (compiled, profile) = CompiledSchema::compile_profiled(
            br#"{"const":"done"}"#,
            &vocabulary,
            &CompileOptions::default(),
        )
        .unwrap();
        let accounted = profile.schema_parse_ns
            + profile.normalization_ns
            + profile.grammar_lowering_ns
            + profile.grammar_reduction_ns
            + profile.scc_analysis_ns
            + profile.regular_certification_ns
            + profile.nfa_construction_ns
            + profile.dfa_determinization_ns
            + profile.vocabulary_projection_ns
            + profile.residual_grammar_ns
            + profile.lalr_construction_ns
            + profile.earley_preparation_ns
            + profile.vocabulary_trie_ns;

        assert_eq!(compiled.report.selected_backend, BackendKind::WholeDfa);
        assert!(compiled.vocabulary.is_none());
        assert_eq!(profile.vocabulary_trie_ns, 0);
        assert!(profile.total_ns >= accounted);
    }

    #[test]
    fn whole_dfa_guides_share_the_compiled_index() {
        let vocabulary = vocabulary(&[(r#""done""#, 0)], 1);
        let compiled = CompiledSchema::compile(
            br#"{"const":"done"}"#,
            &vocabulary,
            &CompileOptions::default(),
        )
        .unwrap();
        let first = compiled.guide(1, RuntimeLimits::default()).unwrap();
        let second = compiled.guide(1, RuntimeLimits::default()).unwrap();

        match (&first.runtime, &second.runtime) {
            (
                GuideRuntime::Dfa {
                    index: first_index, ..
                },
                GuideRuntime::Dfa {
                    index: second_index,
                    ..
                },
            ) => assert!(Arc::ptr_eq(first_index, second_index)),
            _ => panic!("expected whole-DFA guides"),
        }
    }

    #[test]
    fn structural_profile_does_not_claim_automaton_or_projection_work() {
        let schema = include_bytes!("../testdata/regressions/recursive_optional_property.json");
        let vocabulary = vocabulary(&[("null", 0)], 1);
        let (compiled, profile) =
            CompiledSchema::compile_profiled(schema, &vocabulary, &CompileOptions::default())
                .unwrap();

        assert!(matches!(
            compiled.report.selected_backend,
            BackendKind::Lalr | BackendKind::Earley
        ));
        assert_eq!(profile.nfa_construction_ns, 0);
        assert_eq!(profile.dfa_determinization_ns, 0);
        assert_eq!(profile.vocabulary_projection_ns, 0);
    }

    #[test]
    fn recursive_object_recognizes_depth_one_hundred() {
        let schema = include_bytes!("../testdata/regressions/recursive_optional_property.json");
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        let mut value = String::from(r#"{"value":"x"}"#);
        for _ in 1..100 {
            value = format!(r#"{{"next":{value},"value":"x"}}"#);
        }
        assert!(matches!(
            compiled.report.selected_backend,
            BackendKind::Lalr | BackendKind::Earley
        ));
        assert!(accepts(&compiled, value.as_bytes()));
        assert!(!accepts(&compiled, br#"{"next":{"value":"x"}}"#));

        let mut truncated = compiled.recognizer(RuntimeLimits::default()).unwrap();
        assert_eq!(
            truncated
                .try_advance_bytes(&value.as_bytes()[..value.len() - 1])
                .unwrap(),
            Advance::Live
        );
        assert!(truncated.is_live());
        assert!(!truncated.is_accepting());

        let mut recognizer = compiled.recognizer(RuntimeLimits::default()).unwrap();
        assert_eq!(
            recognizer.try_advance_bytes(br#"{"value":"x"}"#).unwrap(),
            Advance::Accepting
        );
        assert_eq!(
            recognizer.try_advance_bytes(b"}").unwrap(),
            Advance::Rejected
        );
        assert!(recognizer.is_accepting());
    }

    #[test]
    fn recursive_array_recognizes_depth_one_hundred() {
        let schema = br##"{
          "$defs":{"node":{"anyOf":[{"type":"null"},{"type":"array","prefixItems":[{"$ref":"#/$defs/node"}],"items":false}]}},
          "$ref":"#/$defs/node"
        }"##;
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        let value = format!("{}null{}", "[".repeat(100), "]".repeat(100));
        assert!(accepts(&compiled, value.as_bytes()));
        assert!(!accepts(
            &compiled,
            format!("{}null{}", "[".repeat(100), "]".repeat(99)).as_bytes()
        ));
    }

    #[test]
    fn forced_backends_expose_conflicts_and_agree() {
        let schema = br#"{"anyOf":[{"const":"x"},{"const":"x"}]}"#;
        let vocabulary = vocabulary(&[(r#""x""#, 0)], 1);
        let error = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary,
            &CompileOptions::default(),
            BackendPolicy::ForceLalr,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::Compile(CompileError::BackendUnavailable {
                backend: "lalr",
                ..
            })
        ));

        let dfa = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary,
            &CompileOptions::default(),
            BackendPolicy::ForceDfa,
        )
        .unwrap();
        let earley = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary,
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        assert!(accepts(&dfa, br#""x""#));
        assert!(accepts(&earley, br#""x""#));
        assert!(!accepts(&dfa, br#""y""#));
        assert!(!accepts(&earley, br#""y""#));
    }

    #[test]
    fn byte_chunks_and_checkpoint_replay_are_equivalent() {
        let compiled = CompiledSchema::compile(
            r#"{"const":"é"}"#.as_bytes(),
            &vocabulary(&[(r#""é""#, 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        let bytes = "\"é\"".as_bytes();
        let mut whole = compiled.recognizer(RuntimeLimits::default()).unwrap();
        assert_eq!(whole.try_advance_bytes(bytes).unwrap(), Advance::Accepting);
        for split in 0..=bytes.len() {
            let mut chunked = compiled.recognizer(RuntimeLimits::default()).unwrap();
            chunked.try_advance_bytes(&bytes[..split]).unwrap();
            let checkpoint = chunked.checkpoint();
            chunked.try_advance_bytes(&bytes[split..]).unwrap();
            assert!(chunked.is_accepting());
            chunked.restore(&checkpoint).unwrap();
            chunked.try_advance_bytes(&bytes[split..]).unwrap();
            assert!(chunked.is_accepting());
        }
    }

    #[test]
    fn rejected_and_limited_advances_are_transactional() {
        let compiled = CompiledSchema::compile(
            br#"{"const":"ok"}"#,
            &vocabulary(&[(r#""ok""#, 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        let mut recognizer = compiled.recognizer(RuntimeLimits::default()).unwrap();
        let before = recognizer.checkpoint();
        assert_eq!(
            recognizer.try_advance_bytes(b"bad").unwrap(),
            Advance::Rejected
        );
        assert_eq!(recognizer.checkpoint(), before);

        let limits = RuntimeLimits {
            max_input_bytes: 1,
            ..RuntimeLimits::default()
        };
        let mut limited = compiled.recognizer(limits).unwrap();
        let before = limited.checkpoint();
        assert!(matches!(
            limited.try_advance_bytes(b"\"o"),
            Err(crate::error::RuntimeError::ResourceLimitExceeded { .. })
        ));
        assert_eq!(limited.checkpoint(), before);
    }

    #[test]
    fn parser_construction_budget_falls_back_without_weakening() {
        let schema = include_bytes!("../testdata/regressions/recursive_array.json");
        let options = CompileOptions {
            limits: crate::schema::CompileLimits {
                max_lr_states: 1,
                ..crate::schema::CompileLimits::default()
            },
        };
        let vocabulary = vocabulary(&[("unused", 0)], 1);
        let compiled = CompiledSchema::compile(schema, &vocabulary, &options).unwrap();
        assert_eq!(compiled.report.selected_backend, BackendKind::Earley);
        assert!(accepts(&compiled, b"[[null]]"));
        assert!(matches!(
            CompiledSchema::compile_with_policy(
                schema,
                &vocabulary,
                &options,
                BackendPolicy::ForceLalr,
            ),
            Err(crate::Error::Compile(CompileError::ResourceLimitExceeded {
                stage: crate::error::CompileStage::LrConstruction,
                ..
            }))
        ));
    }

    #[test]
    fn ambiguous_structural_grammar_reports_provenance_and_uses_earley() {
        let schema = include_bytes!("../testdata/regressions/ambiguous_anyof.json");
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        assert_eq!(compiled.report.selected_backend, BackendKind::Earley);
        assert!(compiled.report.conflicts >= 1);
        assert!(compiled
            .report
            .conflict_summaries
            .iter()
            .all(|conflict| !conflict.provenance.is_empty()));
        assert!(accepts(&compiled, b"[[null]]"));
    }

    #[test]
    fn lexical_ambiguity_prevents_lalr_selection() {
        let schema = br##"{
          "$defs":{"node":{"anyOf":[
            {"const":null},
            {"type":"array","prefixItems":[
              {"$ref":"#/$defs/node"},{"type":"integer"}
            ],"items":false}
          ]}},
          "$ref":"#/$defs/node"
        }"##;
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();

        assert_eq!(compiled.report.selected_backend, BackendKind::Earley);
        assert_eq!(compiled.report.conflicts, 0);
        assert!(accepts(&compiled, b"[null,12]"));
        assert!(!accepts(&compiled, b"[null,]"));
    }

    #[test]
    fn earley_runtime_limit_is_typed_during_initial_saturation() {
        let schema = include_bytes!("../testdata/regressions/recursive_array.json");
        let compiled = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        let limits = RuntimeLimits {
            max_items_per_column: 1,
            ..RuntimeLimits::default()
        };
        assert!(matches!(
            compiled.recognizer(limits),
            Err(crate::error::RuntimeError::ResourceLimitExceeded {
                resource: crate::error::RuntimeResource::ItemsPerColumn,
                ..
            })
        ));
    }

    #[test]
    fn checkpoints_are_bound_to_one_recognizer() {
        let compiled = CompiledSchema::compile(
            br#"{"const":"x"}"#,
            &vocabulary(&[(r#""x""#, 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        let first = compiled.recognizer(RuntimeLimits::default()).unwrap();
        let checkpoint = first.checkpoint();
        let mut second = compiled.recognizer(RuntimeLimits::default()).unwrap();
        assert!(matches!(
            second.restore(&checkpoint),
            Err(crate::error::RuntimeError::InvalidCheckpoint { .. })
        ));

        for (schema, policy) in [
            (
                include_bytes!("../testdata/regressions/recursive_array.json").as_slice(),
                BackendPolicy::ForceLalr,
            ),
            (
                include_bytes!("../testdata/regressions/ambiguous_anyof.json").as_slice(),
                BackendPolicy::ForceEarley,
            ),
        ] {
            let compiled = CompiledSchema::compile_with_policy(
                schema,
                &vocabulary(&[("unused", 0)], 1),
                &CompileOptions::default(),
                policy,
            )
            .unwrap();
            let first = compiled.recognizer(RuntimeLimits::default()).unwrap();
            let checkpoint = first.checkpoint();
            let mut second = compiled.recognizer(RuntimeLimits::default()).unwrap();
            assert!(matches!(
                second.restore(&checkpoint),
                Err(crate::error::RuntimeError::InvalidCheckpoint { .. })
            ));
        }
    }

    #[test]
    fn earley_checkpoint_replay_restores_stats_and_state() {
        let compiled = CompiledSchema::compile_with_policy(
            include_bytes!("../testdata/regressions/ambiguous_anyof.json"),
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        let mut recognizer = compiled.recognizer(RuntimeLimits::default()).unwrap();
        assert_eq!(recognizer.try_advance_bytes(b"[[").unwrap(), Advance::Live);
        let checkpoint = recognizer.checkpoint();
        assert_eq!(
            recognizer.try_advance_bytes(b"null]]").unwrap(),
            Advance::Accepting
        );
        let finished = recognizer.checkpoint();
        recognizer.restore(&checkpoint).unwrap();
        assert_eq!(
            recognizer.try_advance_bytes(b"null]]").unwrap(),
            Advance::Accepting
        );
        assert_eq!(recognizer.checkpoint(), finished);
    }

    #[test]
    fn dfa_lalr_and_earley_agree_on_every_prefix() {
        let schema = br#"{"const":"x"}"#;
        let vocabulary = vocabulary(&[(r#""x""#, 0)], 1);
        let compiled = [
            BackendPolicy::ForceDfa,
            BackendPolicy::ForceLalr,
            BackendPolicy::ForceEarley,
        ]
        .map(|policy| {
            CompiledSchema::compile_with_policy(
                schema,
                &vocabulary,
                &CompileOptions::default(),
                policy,
            )
            .unwrap()
        });
        for input in [
            br#""x""#.as_slice(),
            br#""y""#.as_slice(),
            b"\"x".as_slice(),
        ] {
            let mut traces = Vec::new();
            for compiled in &compiled {
                let mut recognizer = compiled.recognizer(RuntimeLimits::default()).unwrap();
                let mut trace = vec![(recognizer.is_live(), recognizer.is_accepting())];
                for byte in input {
                    let advance = recognizer.try_advance_bytes(&[*byte]).unwrap();
                    trace.push((recognizer.is_live(), recognizer.is_accepting()));
                    if advance == Advance::Rejected {
                        break;
                    }
                }
                traces.push(trace);
            }
            assert_eq!(traces[0], traces[1]);
            assert_eq!(traces[0], traces[2]);
        }
    }

    #[test]
    fn structural_whole_buffer_and_chunk_boundaries_agree() {
        let schema = include_bytes!("../testdata/regressions/recursive_optional_property.json");
        let compiled = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary(&[("unused", 0)], 1),
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        let input = r#"{"next":{"value":"x"},"value":"é"}"#.as_bytes();
        let mut whole = compiled.recognizer(RuntimeLimits::default()).unwrap();
        assert_eq!(whole.try_advance_bytes(input).unwrap(), Advance::Accepting);
        for split in 0..=input.len() {
            let mut chunked = compiled.recognizer(RuntimeLimits::default()).unwrap();
            let first = chunked.try_advance_bytes(&input[..split]).unwrap();
            assert_ne!(first, Advance::Rejected);
            let second = chunked.try_advance_bytes(&input[split..]).unwrap();
            assert_eq!(second, Advance::Accepting, "split {split}");
            assert_eq!(chunked.is_live(), whole.is_live());
            assert_eq!(chunked.is_accepting(), whole.is_accepting());
        }
    }

    #[test]
    fn structural_guide_mask_matches_brute_force_with_sparse_aliases() {
        let schema = include_bytes!("../testdata/regressions/ambiguous_anyof.json");
        let mut vocabulary = vocabulary(
            &[("[", 0), ("[[", 1), ("null", 4), ("[[null]]", 5), ("x", 6)],
            100,
        );
        vocabulary.try_insert("null", 19).unwrap();
        let compiled = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary,
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        let mut guide = compiled.guide(8, RuntimeLimits::default()).unwrap();
        let parser_before = guide.parser_stats();
        let mut mask = vec![u32::MAX; guide.mask_words()];
        guide.fill_mask(&mut mask).unwrap();
        assert_eq!(guide.vocab_size(), 101);
        assert_eq!(guide.mask_words(), 4);
        assert_eq!(guide.parser_stats(), parser_before);
        for _ in 0..100 {
            let mut repeated = vec![u32::MAX; guide.mask_words()];
            guide.fill_mask(&mut repeated).unwrap();
            assert_eq!(repeated, mask);
        }
        assert_eq!(guide.parser_stats(), parser_before);
        assert_eq!(guide.mask_stats().cache_misses, 1);
        assert_eq!(guide.mask_stats().cache_hits, 100);
        assert_eq!(guide.mask_stats().cache_bytes, 16);

        for token_id in [0, 1, 4, 5, 6, 19, 100] {
            let expected = if token_id == 100 {
                false
            } else {
                let bytes = compiled
                    .vocabulary
                    .as_ref()
                    .unwrap()
                    .shared_bytes(token_id)
                    .unwrap_or_else(|| Arc::from([]));
                let mut recognizer = compiled.recognizer(RuntimeLimits::default()).unwrap();
                recognizer.try_advance_bytes(&bytes).unwrap() != Advance::Rejected
                    && recognizer.is_live()
            };
            let actual = mask[token_id as usize / 32] & (1 << (token_id % 32)) != 0;
            assert_eq!(actual, expected, "token {token_id}");
        }
        assert_eq!(mask[0] & (1 << 4) != 0, mask[0] & (1 << 19) != 0);
    }

    #[test]
    fn structural_guide_commit_eos_rollback_and_reset_are_exact() {
        let schema = include_bytes!("../testdata/regressions/ambiguous_anyof.json");
        let compiled = CompiledSchema::compile_with_policy(
            schema,
            &vocabulary(&[("[[null]]", 5), ("x", 6)], 100),
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        let mut guide = compiled.guide(8, RuntimeLimits::default()).unwrap();
        let initial_fingerprint = guide.state_fingerprint();
        assert!(matches!(
            guide.advance(6),
            Err(crate::error::RuntimeError::TokenNotAllowed { token_id: 6 })
        ));
        assert_eq!(guide.get_allowed_rollback(), 0);
        assert_eq!(guide.state_fingerprint(), initial_fingerprint);

        guide.advance(5).unwrap();
        assert_eq!(guide.mask_stats().cache_bytes, 0);
        assert!(guide.is_accepting());
        assert!(!guide.is_finished());
        assert_eq!(guide.get_tokens().unwrap(), vec![100]);
        guide.advance(100).unwrap();
        assert!(guide.is_finished());
        assert!(!guide.is_accepting());
        assert!(guide.get_tokens().unwrap().is_empty());

        guide.rollback(1).unwrap();
        assert!(!guide.is_finished());
        assert!(guide.is_accepting());
        guide.rollback(1).unwrap();
        assert_eq!(guide.state_fingerprint(), initial_fingerprint);
        assert_eq!(guide.position(), 0);

        guide.advance(5).unwrap();
        guide.reset().unwrap();
        assert_eq!(guide.get_allowed_rollback(), 0);
        assert_eq!(guide.position(), 0);
        assert_eq!(guide.state_fingerprint(), initial_fingerprint);
    }

    #[test]
    fn zero_rollback_limit_never_retains_history() {
        let compiled = CompiledSchema::compile(
            br#"{"const":"x"}"#,
            &vocabulary(&[(r#""x""#, 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();
        let mut guide = compiled.guide(0, RuntimeLimits::default()).unwrap();
        guide.advance(0).unwrap();
        assert_eq!(guide.get_allowed_rollback(), 0);
        assert!(matches!(
            guide.rollback(1),
            Err(crate::error::RuntimeError::RollbackUnavailable {
                requested: 1,
                available: 0
            })
        ));
    }

    #[test]
    fn mask_limit_error_restores_structural_parser_state() {
        let compiled = CompiledSchema::compile_with_policy(
            include_bytes!("../testdata/regressions/ambiguous_anyof.json"),
            &vocabulary(&[("null", 0)], 1),
            &CompileOptions::default(),
            BackendPolicy::ForceEarley,
        )
        .unwrap();
        let limits = RuntimeLimits {
            max_mask_trie_edges: 0,
            ..RuntimeLimits::default()
        };
        let mut guide = compiled.guide(8, limits).unwrap();
        let before = guide.parser_stats();
        let fingerprint = guide.state_fingerprint();
        let mut mask = vec![0; guide.mask_words()];
        assert!(matches!(
            guide.fill_mask(&mut mask),
            Err(crate::error::RuntimeError::ResourceLimitExceeded {
                resource: crate::error::RuntimeResource::MaskTraversal,
                ..
            })
        ));
        assert_eq!(guide.parser_stats(), before);
        assert_eq!(guide.state_fingerprint(), fingerprint);
    }
}
