# Provenance and attribution

OC-Earley began from [dottxt-ai/outlines-core](https://github.com/dottxt-ai/outlines-core) commit `116a7c381b446814fdfc9b2b28bea7ca4f900aae`, imported as repository root commit `57188c865dfcbf6dbf9c75118e5c3693bb02e1d8`. The upstream work is licensed under Apache License 2.0.

The following inherited files have been modified or relocated. Each modified file carries a notice referring back to this record.

| OC-Earley file | Upstream origin | Material change |
|---|---|---|
| `src/error.rs` | `src/error.rs` | Extended errors for schema compilation and structural recognition. |
| `src/index.rs` | `src/index.rs` | Generalized byte-DFA projection while retaining the token-index interface. |
| `src/json_schema/mod.rs` | `src/json_schema/mod.rs` | Integrated the strict schema compiler with compatibility entry points. |
| `src/lib.rs` | `src/lib.rs` | Renamed the crate and exposed schema, grammar, and recognizer APIs. |
| `src/prelude.rs` | `src/prelude.rs` | Updated exports for the renamed and extended API. |
| `src/python_bindings/mod.rs` | `src/python_bindings/mod.rs` | Renamed the extension and added compiled-schema and structural-guide bindings. |
| `src/vocabulary/mod.rs` | `src/vocabulary/mod.rs` | Added validated vocabulary-width and reverse-token support. |
| `src/bin/convert-json-schema.rs` | `src/bin/convert-json-schema.rs` | Renamed package references while retaining the compatibility command. |
| `oc_earley/__init__.py` | `outlines_core/__init__.py` | Renamed the package and exported the compiled-schema API. |
| `oc_earley/_json_schema.py` | `outlines_core/_json_schema.py` | Renamed imports and connected the schema compiler. |
| `oc_earley/kernels/mlx.py` | `outlines_core/kernels/mlx.py` | Renamed imports while retaining the packed-mask integration. |
| `oc_earley/kernels/numpy.py` | `outlines_core/kernels/numpy.py` | Renamed imports while retaining the packed-mask integration. |
| `oc_earley/kernels/torch.py` | `outlines_core/kernels/torch.py` | Renamed imports while retaining the packed-mask integration. |
| `oc_earley/kernels/__init__.py` | `outlines_core/kernels/__init__.py` | Relocated without source changes. |
| `tests/test_guide.py` | `tests/test_guide.py` | Updated identity and extended guide-state coverage. |
| `tests/test_imports.py` | `tests/test_imports.py` | Updated import and serialization identity checks. |
| `tests/test_index.py` | `tests/test_index.py` | Extended token-index and projection coverage. |
| `tests/test_json_schema.py` | `tests/test_json_schema.py` | Updated identity while retaining compatibility coverage. |
| `tests/test_kernels.py` | `tests/test_kernels.py` | Updated imports and packed-mask checks. |
| `tests/test_statistical.py` | `tests/test_statistical.py` | Updated package identity. |
| `tests/test_vocabulary.py` | `tests/test_vocabulary.py` | Extended vocabulary validation coverage. |
| `.gitignore` | `.gitignore` | Updated generated-artifact exclusions for the fork. |
| `Cargo.toml` | `Cargo.toml` | Renamed crate metadata and added compiler/runtime dependencies and packaging rules. |
| `environment.yml` | `environment.yml` | Renamed the development environment. |
| `Makefile` | `Makefile` | Updated package names and development commands. |
| `pyproject.toml` | `pyproject.toml` | Renamed Python metadata and configured mixed Rust/Python packaging. |
| `README.md` | `README.md` | Rewritten for OC-Earley's architecture, API, and attribution. |
| `rustfmt.toml` | `rustfmt.toml` | Retained formatting policy with project attribution. |
| `uv.lock` | `uv.lock` | Regenerated for the renamed package and dependency set. |
| `benchmarks/asv.conf.json` | `benchmarks/asv.conf.json` | Renamed benchmark metadata and commands. |
| `benchmarks/bench_json_schema.py` | `benchmarks/bench_json_schema.py` | Updated imports and schema benchmark setup. |
| `benchmarks/bench_kernels.py` | `benchmarks/bench_kernels.py` | Updated imports for the renamed kernels. |
| `benchmarks/bench_regex_guide.py` | `benchmarks/bench_regex_guide.py` | Updated imports and guide benchmark setup. |
| `benchmarks/bench_torch_e2e.py` | `benchmarks/bench_torch_e2e.py` | Updated imports and end-to-end benchmark setup. |
| `.github/workflows/publish.yml` | Upstream release workflows | Reworked release verification, wheel building, and registry publication. |
| `.github/workflows/release-dry-run.yml` | `.github/workflows/dry_run_publish.yml` and related build workflows | Reworked local-equivalent release checks without publication. |

The unmodified inherited source files retain their original form and remain covered by the upstream license and copyright notice. New files not listed above are original OC-Earley work.

See [LICENSE](LICENSE) for the Apache License 2.0 terms and [NOTICE](NOTICE) for the required attribution and summary of material modifications. OC-Earley is an independent project and is not endorsed by the upstream authors.
