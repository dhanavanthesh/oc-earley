use std::path::PathBuf;

use oc_earley::engine::{BackendKind, SccReport, TierReport};
use oc_earley::grammar::{CertificateFailureReason, RegularCertificateKind};
use oc_earley::{CompileError, CompileOptions, CompiledSchema};

const EXIT_INTERNAL: i32 = 1;
const EXIT_SCHEMA: i32 = 2;
const EXIT_RESOURCE: i32 = 3;
const EXIT_STRUCTURAL: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    TierReport {
        schema: PathBuf,
        format: OutputFormat,
    },
    Compile {
        schema: PathBuf,
        format: OutputFormat,
    },
    Help,
}

fn main() {
    std::process::exit(run(std::env::args().skip(1).collect()));
}

fn run(arguments: Vec<String>) -> i32 {
    let command = match parse_command(&arguments) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("{}", usage());
            return EXIT_SCHEMA;
        }
    };
    let (path, format) = match command {
        Command::TierReport { schema, format } | Command::Compile { schema, format } => {
            (schema, format)
        }
        Command::Help => {
            println!("{}", usage());
            return 0;
        }
    };
    let schema = match std::fs::read(&path) {
        Ok(schema) => schema,
        Err(error) => {
            eprintln!("error: cannot read {}: {error}", path.display());
            return EXIT_SCHEMA;
        }
    };
    let report = match CompiledSchema::analyze(&schema, &CompileOptions::default()) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("error: {error}");
            return error_exit_code(&error);
        }
    };
    if let Err(message) = write_report(&report, format) {
        eprintln!("error: {message}");
        return EXIT_INTERNAL;
    }
    0
}

fn parse_command(arguments: &[String]) -> Result<Command, String> {
    let Some(name) = arguments.first().map(String::as_str) else {
        return Err("missing command".to_owned());
    };
    if matches!(name, "help" | "--help" | "-h") {
        return if arguments.len() == 1 {
            Ok(Command::Help)
        } else {
            Err("help does not accept arguments".to_owned())
        };
    }
    if !matches!(name, "tier-report" | "compile") {
        return Err(format!("unknown command `{name}`"));
    }
    let schema = arguments
        .get(1)
        .filter(|argument| !argument.starts_with('-'))
        .ok_or_else(|| format!("`{name}` requires a schema path"))?;
    let mut format = OutputFormat::Text;
    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--format" => {
                let value = arguments
                    .get(index + 1)
                    .ok_or_else(|| "`--format` requires `text` or `json`".to_owned())?;
                format = match value.as_str() {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    _ => return Err(format!("unsupported report format `{value}`")),
                };
                index += 2;
            }
            argument => return Err(format!("unexpected argument `{argument}`")),
        }
    }
    let schema = PathBuf::from(schema);
    Ok(if name == "tier-report" {
        Command::TierReport { schema, format }
    } else {
        Command::Compile { schema, format }
    })
}

fn write_report(report: &TierReport, format: OutputFormat) -> Result<(), String> {
    match format {
        OutputFormat::Json => {
            let rendered =
                serde_json::to_string_pretty(report).map_err(|error| error.to_string())?;
            println!("{rendered}");
        }
        OutputFormat::Text => print!("{}", render_text_report(report)),
    }
    Ok(())
}

fn render_text_report(report: &TierReport) -> String {
    let mut output = String::new();
    push_line(&mut output, "dialect", &report.dialect);
    push_line(&mut output, "profile", &report.profile);
    push_line(&mut output, "canonical-policy", &report.canonical_policy);
    push_line(
        &mut output,
        "backend",
        match report.selected_backend {
            BackendKind::WholeDfa => "whole-dfa",
            BackendKind::Lalr => "lalr",
            BackendKind::Earley => "earley",
        },
    );
    push_count(&mut output, "schema-nodes", report.schema_nodes);
    push_count(&mut output, "reference-edges", report.reference_edges);
    push_count(&mut output, "grammar-symbols", report.grammar_symbols);
    push_count(
        &mut output,
        "grammar-productions",
        report.grammar_productions,
    );
    push_optional_count(&mut output, "nfa-states", report.nfa_states);
    push_optional_count(&mut output, "nfa-transitions", report.nfa_transitions);
    push_optional_count(&mut output, "dfa-states", report.dfa_states);
    push_optional_count(&mut output, "dfa-bytes", report.dfa_bytes);
    push_count(
        &mut output,
        "collapsed-regular-regions",
        report.collapsed_regular_regions,
    );
    push_count(&mut output, "compiled-terminals", report.compiled_terminals);
    push_count(
        &mut output,
        "terminal-dfa-states",
        report.terminal_dfa_states,
    );
    push_count(&mut output, "terminal-dfa-bytes", report.terminal_dfa_bytes);
    push_count(
        &mut output,
        "canonical-lr-states",
        report.canonical_lr_states,
    );
    push_count(&mut output, "lalr-states", report.lalr_states);
    push_count(&mut output, "action-entries", report.action_entries);
    push_count(&mut output, "goto-entries", report.goto_entries);
    push_count(&mut output, "conflicts", report.conflicts);
    push_count(
        &mut output,
        "nullable-nonterminals",
        report.nullable_nonterminals,
    );
    push_count(&mut output, "leo-eligible-rules", report.leo_eligible_rules);
    for scc in &report.sccs {
        render_scc(&mut output, scc);
    }
    for diagnostic in &report.diagnostics {
        output.push_str("diagnostic ");
        output.push_str(&diagnostic.code);
        output.push_str(" at ");
        output.push_str(&diagnostic.location.pointer.to_string());
        output.push_str(": ");
        output.push_str(&diagnostic.message);
        output.push('\n');
    }
    output
}

fn render_scc(output: &mut String, report: &SccReport) {
    output.push_str("scc ");
    output.push_str(&report.id.to_string());
    output.push_str(": members=");
    for (index, member) in report.members.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push_str(&member.to_string());
    }
    output.push_str(" certificate=");
    output.push_str(match report.certificate {
        Some(RegularCertificateKind::Acyclic) => "acyclic",
        Some(RegularCertificateKind::RightLinear) => "right-linear",
        Some(RegularCertificateKind::LeftLinear) => "left-linear",
        None => "none",
    });
    if let Some(reason) = report.failure_reason {
        output.push_str(" failure=");
        output.push_str(match reason {
            CertificateFailureReason::MultipleRecursiveOccurrences => {
                "multiple-recursive-occurrences"
            }
            CertificateFailureReason::MixedLinearOrientation => "mixed-linear-orientation",
            CertificateFailureReason::RecursiveSymbolInInterior => "recursive-symbol-in-interior",
            CertificateFailureReason::UncertifiedDependency => "uncertified-dependency",
            CertificateFailureReason::UnsupportedRegularOperation => {
                "unsupported-regular-operation"
            }
        });
    }
    output.push_str(" pointers=");
    for (index, pointer) in report.source_pointers.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push_str(&pointer.to_string());
    }
    output.push('\n');
}

fn push_line(output: &mut String, name: &str, value: &str) {
    output.push_str(name);
    output.push_str(": ");
    output.push_str(value);
    output.push('\n');
}

fn push_count(output: &mut String, name: &str, value: usize) {
    push_line(output, name, &value.to_string());
}

fn push_optional_count(output: &mut String, name: &str, value: Option<usize>) {
    match value {
        Some(value) => push_count(output, name, value),
        None => push_line(output, name, "not-applicable"),
    }
}

fn error_exit_code(error: &CompileError) -> i32 {
    match error {
        CompileError::InternalInvariant { .. } => EXIT_INTERNAL,
        CompileError::ResourceLimitExceeded { .. } => EXIT_RESOURCE,
        CompileError::StructuralBackendRequired { .. } => EXIT_STRUCTURAL,
        _ => EXIT_SCHEMA,
    }
}

fn usage() -> &'static str {
    "Usage:\n  oc-earley tier-report <schema.json> [--format text|json]\n  oc-earley compile <schema.json> [--format text|json]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_commands_strictly() {
        assert_eq!(
            parse_command(&[
                "tier-report".to_owned(),
                "schema.json".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ]),
            Ok(Command::TierReport {
                schema: PathBuf::from("schema.json"),
                format: OutputFormat::Json,
            })
        );
        assert!(parse_command(&["compile".to_owned()]).is_err());
        assert!(parse_command(&[
            "tier-report".to_owned(),
            "schema.json".to_owned(),
            "--unknown".to_owned(),
        ])
        .is_err());
    }

    #[test]
    fn text_report_is_deterministic() {
        let schema = include_bytes!("../../testdata/regressions/deep_acyclic_ref.json");
        let report = CompiledSchema::analyze(schema, &CompileOptions::default()).unwrap();
        assert_eq!(render_text_report(&report), render_text_report(&report));
        assert!(render_text_report(&report).contains("backend: whole-dfa\n"));
    }

    #[test]
    fn compiler_errors_have_stable_exit_classes() {
        assert_eq!(
            error_exit_code(&CompileError::ResourceLimitExceeded {
                stage: oc_earley::CompileStage::DfaDeterminization,
                observed: 2,
                limit: 1,
            }),
            EXIT_RESOURCE
        );
        assert_eq!(
            error_exit_code(&CompileError::InternalInvariant { message: "test" }),
            EXIT_INTERNAL
        );
        assert_eq!(
            error_exit_code(&CompileError::InvalidJson {
                message: "test".to_owned(),
            }),
            EXIT_SCHEMA
        );
    }
}
