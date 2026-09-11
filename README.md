# OC-Earley

OC-Earley is a correctness-first constrained-decoding compiler for JSON Schema Draft 2020-12. It preserves the existing packed vocabulary-mask ABI from outlines-core while replacing depth-capped schema expansion with graph-based compilation.

The governing rule is simple: exact or explicit. If the compiler cannot prove that a transformation preserves the canonical schema language, it reports an unsupported construct or selects a structural backend. It never drops an assertion or property silently.

## Compiler pipeline

The schema compiler follows this pipeline:

```text
JSON Schema
  -> strict K1 validation
  -> local-reference graph
  -> exact normalization
  -> provenance-carrying grammar
  -> reduction and SCC analysis
  -> exact regular certificate
  -> byte DFA
  -> vocabulary Index
  -> packed u32 mask
```

Schemas that are not certified regular produce `StructuralBackendRequired`. Structural recognizers are not shipped in the current release, and the library never substitutes a weaker approximation.

## Identities

- Cargo package and CLI: `oc-earley`
- Rust crate: `oc_earley`
- Python distribution: `oc-earley`
- Python import: `oc_earley`

## Legacy regular-expression API

```python
from oc_earley import Guide, Index, Vocabulary

vocabulary = Vocabulary(3, {"0": [0], "1": [1]})
index = Index(r"0|[1-9][0-9]*", vocabulary)
guide = Guide(index)
```

`Guide.write_mask_into(data_ptr, numel, element_size)` keeps the inherited ABI. The buffer is zeroed first, each token uses `word = id / 32` and `bit = id % 32`, and bits are LSB-first in native-endian `u32` words.

## Attribution

OC-Earley is a modified fork of `dottxt-ai/outlines-core`. The original Apache License 2.0 text is retained verbatim in `LICENSE`; material modifications are summarized in `NOTICE`.
