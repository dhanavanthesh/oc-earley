//! Schema compiler orchestration and backend selection.

use serde::Serialize;

use crate::error::CompileError;
use crate::grammar::{
    self, CertificateFailureReason, CompiledByteDfa, Grammar, NonterminalId,
    RegularCertificateKind, SccId,
};
use crate::index::Index;
use crate::schema::{
    CompileOptions, DiagnosticLocation, SchemaArena, SchemaPointer, CANONICAL_POLICY_ID,
    COMPILED_FORMAT_VERSION, DRAFT_2020_12, PROFILE_ID,
};
use crate::vocabulary::Vocabulary;
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    WholeDfa,
    StructuralPending,
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
    pub dfa_states: Option<usize>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SemanticContract {
    pub format_version: u32,
    pub dialect: String,
    pub profile: String,
    pub canonical_policy: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralPlan {
    pub location: DiagnosticLocation,
    pub failed_sccs: Vec<SccId>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompiledBackend {
    Dfa(Index),
    StructuralPending(StructuralPlan),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledSchema {
    pub semantic_contract: SemanticContract,
    pub normalized: SchemaArena,
    pub grammar: Grammar,
    pub report: TierReport,
    pub backend: CompiledBackend,
}

struct PreparedCompilation {
    arena: SchemaArena,
    grammar: Grammar,
    report: TierReport,
    dfa: Option<CompiledByteDfa>,
    structural_plan: StructuralPlan,
}

impl CompiledSchema {
    pub fn analyze(schema: &[u8], options: &CompileOptions) -> Result<TierReport, CompileError> {
        Ok(prepare(schema, options)?.report)
    }

    pub fn compile(
        schema: &[u8],
        vocabulary: &Vocabulary,
        options: &CompileOptions,
    ) -> Result<Self> {
        let prepared = prepare(schema, options)?;
        let backend = match prepared.dfa {
            Some(dfa) => CompiledBackend::Dfa(Index::from_certified_dfa(&dfa, vocabulary)?),
            None => CompiledBackend::StructuralPending(prepared.structural_plan),
        };
        Ok(Self {
            semantic_contract: semantic_contract(),
            normalized: prepared.arena,
            grammar: prepared.grammar,
            report: prepared.report,
            backend,
        })
    }

    pub fn index(&self) -> Result<&Index, CompileError> {
        match &self.backend {
            CompiledBackend::Dfa(index) => Ok(index),
            CompiledBackend::StructuralPending(plan) => {
                Err(CompileError::StructuralBackendRequired {
                    location: plan.location.clone(),
                })
            }
        }
    }
}

fn prepare(schema: &[u8], options: &CompileOptions) -> Result<PreparedCompilation, CompileError> {
    let arena = crate::schema::parse_and_normalize(schema, options)?;
    let mut grammar = grammar::lower(&arena, &options.limits)?;
    grammar::reduce(&mut grammar)?;
    let analysis = grammar::certify_regular(&grammar)?;

    let root = grammar.nonterminals.get(grammar.start as usize).ok_or(
        CompileError::InternalInvariant {
            message: "reduced grammar start does not exist",
        },
    )?;
    let root_location = root.provenance.location();
    let structural_plan = StructuralPlan {
        location: root_location,
        failed_sccs: analysis
            .certificates
            .iter()
            .filter(|certificate| certificate.kind.is_none())
            .map(|certificate| certificate.scc)
            .collect(),
    };
    let dfa = analysis
        .whole_language
        .as_ref()
        .map(|expression| {
            CompiledByteDfa::compile(expression, &options.limits, &root.provenance.pointer)
        })
        .transpose()?;
    let report = build_report(&arena, &grammar, &analysis, dfa.as_ref())?;

    Ok(PreparedCompilation {
        arena,
        grammar,
        report,
        dfa,
        structural_plan,
    })
}

fn build_report(
    arena: &SchemaArena,
    grammar: &Grammar,
    analysis: &grammar::RegularAnalysis,
    dfa: Option<&CompiledByteDfa>,
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
        selected_backend: if dfa.is_some() {
            BackendKind::WholeDfa
        } else {
            BackendKind::StructuralPending
        },
        dfa_states: dfa.map(CompiledByteDfa::state_count),
        diagnostics,
    })
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
    fn recursive_schema_preserves_a_structural_plan() {
        let schema = include_bytes!("../testdata/regressions/recursive_optional_property.json");
        let compiled = CompiledSchema::compile(
            schema,
            &vocabulary(&[("null", 0)], 1),
            &CompileOptions::default(),
        )
        .unwrap();

        assert_eq!(
            compiled.report.selected_backend,
            BackendKind::StructuralPending
        );
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
}
