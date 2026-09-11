<div align="center">

# OC-Earley

Exact recursive JSON Schema recognition for Rust and Python.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

OC-Earley compiles JSON Schema Draft 2020-12 into the least powerful recognizer that can preserve
its canonical JSON language. Regular schemas use a byte DFA. Recursive structure uses LALR(1) when
the parse table and terminal boundaries are deterministic, with Earley as the general fallback.

The compiler never removes an assertion to make a schema compile. Unsupported semantics return a
typed error, and failed finite-state certification selects a structural recognizer.

## Why it exists

A finite automaton cannot remember arbitrary nesting depth. Expanding recursive `$ref` definitions
to a fixed depth only hides that limitation and changes the schema language. OC-Earley keeps the
finite-state path where it is exact and moves genuinely structural regions to a parser.

## Compiler behavior

| Capability | Behavior |
| --- | --- |
| Recursive `$ref` | Resolves local references as a graph without depth-limited expansion |
| Regular schemas | Compiles to the existing byte DFA and vocabulary `Index` |
| Mixed schemas | Collapses certified regular regions into shared terminals |
| LALR(1) | Uses deterministic tables only when parser and lexical conflicts are absent |
| Earley | Recognizes general context-free structure with correct nullable handling |
| Leo | Compresses eligible right-recursive completion chains |
| Transactions | Restores the previous state after rejection or a runtime-limit error |
| Diagnostics | Reports stable schema pointers, conflict provenance, and resource failures |

## Exactness rule

For schema $S$, canonical encoder $\Pi$, and lowering $\Lambda$:

```math
\Lambda(S)=G_S,\qquad L(G_S)=\{\Pi(v)\mid v\models S\}
```

Backend selection preserves that language exactly:

```math
\operatorname{Backend}(G_S)=
\begin{cases}
\mathrm{DFA}, & \text{whole language has an exact regular certificate}\\
\mathrm{LALR}(1), & \text{table and terminal boundaries are conflict-free}\\
\mathrm{Earley+Leo}, & \text{otherwise}
\end{cases}
```

A failed certificate means only that the finite-state compiler could not prove regularity. It does
not classify the language as non-regular and does not permit an approximation.

## Try it from Python

```python
from oc_earley import CompiledSchema, Vocabulary


schema = {
    "$defs": {
        "node": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "prefixItems": [{"$ref": "#/$defs/node"}],
                    "items": False,
                },
            ]
        }
    },
    "$ref": "#/$defs/node",
}

compiled = CompiledSchema.from_json_schema(schema, Vocabulary(1, {"unused": [0]}))
recognizer = compiled.recognizer()

checkpoint = recognizer.checkpoint()
assert recognizer.advance("[[null]]") == "accepting"

recognizer.restore(checkpoint)
assert recognizer.advance("[null]") == "accepting"
```

`advance()` accepts text or bytes and returns `"rejected"`, `"live"`, or `"accepting"`. A rejected
chunk leaves the recognizer unchanged.

## Recognition model

An Earley item records a production prefix recognized between byte positions $i$ and $j$:

```math
[A\rightarrow\alpha\,\bullet\,\beta,i]_j
```

Recognition succeeds only when the augmented start production is complete. Leo changes the amount
of completion work, never the accepted language:

```math
\operatorname{Accept}_{LeoOff}(x)=\operatorname{Accept}_{LeoOn}(x)
```

The packed DFA mask ABI is unchanged:

```math
M[\lfloor id/32\rfloor]\mathrel{|}=1\ll(id\bmod32)
```

Masks use native-endian `u32` words, LSB-first, after clearing the destination buffer.

## Schema coverage

The supported contract includes objects, required properties, arrays, `items`, `prefixItems`, item
bounds, primitive types, `enum`, `const`, local `$ref`, `anyOf`, and supported exact `allOf`
combinations. Unsupported keywords and combinations fail explicitly. `oneOf` is never treated as
union, and `allOf` is never treated as concatenation.

## Runtime notes

The whole-language DFA remains the fastest path. Structural compilation deduplicates literal and
DFA terminals, indexes productions by left-hand side, stores sparse LALR rows, and indexes Earley
waiters by symbol. Construction and recognition have configurable resource limits.

LALR work is proportional to shifts and reductions. Earley retains its classical cubic general
bound and quadratic unambiguous bound; Leo gives linear behavior only for qualifying deterministic
right recursion.

## CLI

```bash
oc-earley tier-report schema.json
oc-earley tier-report schema.json --format json
oc-earley compile schema.json
```

## Building and testing

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --no-default-features
cargo test --all-features
uv run pytest -q
```

## Origin and license

OC-Earley is a modified fork of `dottxt-ai/outlines-core`. The original Apache License 2.0 text is
retained in [LICENSE](LICENSE), and material modifications are summarized in [NOTICE](NOTICE).
