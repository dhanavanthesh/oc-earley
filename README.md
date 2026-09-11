<!-- Portions derived from dottxt-ai/outlines-core and modified by OC-Earley contributors. -->
<!-- See PROVENANCE.md, NOTICE, and LICENSE. -->

<div align="center">

# OC-Earley

**Let JSON nest as deeply as its schema allows.**

Exact recursive JSON Schema recognition and token masking for Rust and Python.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

[The depth trap](#the-depth-trap) · [Language invariant](#language-invariant) ·
[Constrain tokens](#constrain-tokens) · [Parser mechanics](#parser-mechanics) ·
[Useful shapes](#useful-shapes) · [Build](#build)

OC-Earley compiles JSON Schema Draft 2020-12 into the least powerful recognizer that can preserve
its canonical JSON language. Regular schemas use a byte DFA. Recursive structure uses LALR(1) when
the parse table and terminal boundaries are deterministic, with Earley as the general fallback.

The compiler never removes an assertion to make a schema compile. Unsupported semantics return a
typed error, and failed finite-state certification selects a structural recognizer.

## The depth trap

A finite automaton cannot remember arbitrary nesting depth. Expanding recursive `$ref` definitions
to a fixed depth only hides that limitation and changes the schema language. OC-Earley keeps the
finite-state path where it is exact and moves genuinely structural regions to a parser.

## Three machines, one contract

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

## Language invariant

For schema $S$, canonical encoder $\Pi$, and lowering $\Lambda$:

```math
\Lambda(S)=G_S,\qquad L(G_S)=\{\Pi(v)\mid v\models S\}
```

Backend selection preserves that language exactly:

```math
\mathrm{Backend}(G_S)=
\begin{cases}
\mathrm{DFA}, & \text{whole language has an exact regular certificate}\\
\mathrm{LALR}(1), & \text{table and terminal boundaries are conflict-free}\\
\mathrm{Earley+Leo}, & \text{otherwise}
\end{cases}
```

A failed certificate means only that the finite-state compiler could not prove regularity. It does
not classify the language as non-regular and does not permit an approximation.

## Install

Python:

```bash
pip install oc-earley
```

Rust:

```bash
cargo add oc-earley
```

From a checkout:

```bash
uv run maturin develop --release --features python-bindings
```

## Recognize a recursive value

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

## Constrain tokens

Construct the vocabulary with the tokenizer's byte representation and use the guide for token-level
decoding:

```python
from oc_earley import CompiledSchema, Vocabulary
from oc_earley.kernels.numpy import allocate_guide_bitmask, fill_next_token_bitmask


vocabulary = Vocabulary(3, {"[": [0], "null": [1], "]": [2]})
compiled = CompiledSchema.from_json_schema(schema, vocabulary)
guide = compiled.guide(max_rollback=32)
mask = allocate_guide_bitmask(guide)

fill_next_token_bitmask(guide, mask)
guide.advance(0, return_tokens=False)
guide.rollback_state(1)
```

`guide.is_accepting()` means EOS is legal now. `guide.is_finished()` becomes true only after EOS is
committed. Whole-DFA guides retain their numeric state API; structural states are intentionally
opaque.

The same guide contract drives all three backends. Mask queries are pure, rejected tokens leave the
state unchanged, and rollback counts committed tokens rather than parser bytes.

### A model-selected recursive value

A pinned CPU run used Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The guide masked the model logits at every token and
the model selected:

```json
[[[[[[null]]]]]]
```

The run committed EOS, reached depth 6, parsed with `json.loads`, and passed an independent Draft
2020-12 validator. Seed 79 used temperature 1.5 and top-p 1.0. Five earlier attempts produced valid
but shallower values; depth was not inferred from a preconstructed byte string.

## Parser mechanics

An Earley item records a production prefix recognized between byte positions $i$ and $j$:

```math
[A\rightarrow\alpha\,\bullet\,\beta,i]_j
```

Recognition succeeds only when the augmented start production is complete. Leo changes the amount
of completion work, never the accepted language:

```math
\mathrm{Accept}_{LeoOff}(x)=\mathrm{Accept}_{LeoOn}(x)
```

The packed token-mask ABI is unchanged:

```math
M[\lfloor id/32\rfloor]\mathrel{|}=1\ll(id\bmod32)
```

Masks use native-endian `u32` words, LSB-first, after clearing the destination buffer.

For recognizer configuration $h$ and tokenizer byte string $b(v)$, a token bit is exact:

```math
M_h(v)=1\iff\mathrm{Live}(\mathrm{Advance}^{*}(h,b(v)))
```

EOS follows acceptance rather than ordinary byte traversal:

```math
M_h(\mathrm{EOS})=1\iff\mathrm{Accepting}(h)
```

The trie shares token prefixes and prunes rejected subtrees. With $W=\lceil |V|/32\rceil$ mask
words and $E_h$ reachable trie edges, the runtime bound is:

```math
T_{mask}=\Theta(W)+O\!\left(E_h(C_{advance}+C_{checkpoint}+C_{restore})\right)
```

This does not claim constant-time Earley advancement.

## Language surface

The supported contract includes objects, required properties, arrays, `items`, `prefixItems`, item
bounds, primitive types, `enum`, `const`, local `$ref`, `anyOf`, and supported exact `allOf`
combinations. Unsupported keywords and combinations fail explicitly. `oneOf` is never treated as
union, and `allOf` is never treated as concatenation.

`uniqueItems`, `multipleOf`, `format`, remote references, and `unevaluatedProperties` are outside
the current contract and fail explicitly when they would affect the accepted language.

## Cost model

The whole-language DFA remains the fastest path. Structural compilation deduplicates literal and
DFA terminals, indexes productions by left-hand side, stores sparse LALR rows, and indexes Earley
waiters by symbol. Construction and recognition have configurable resource limits.

LALR work is proportional to shifts and reductions. Earley retains its classical cubic general
bound and quadratic unambiguous bound; Leo gives linear behavior only for qualifying deterministic
right recursion.

## Useful shapes

| Shape | Why a finite automaton is insufficient |
| --- | --- |
| Comment threads | Every reply can contain another reply list |
| Syntax trees | Child expressions recursively contain expressions |
| File trees | Directories recursively contain files and directories |
| Organization charts | Reports can contain further reports |
| Execution trees | Each node can expand into nested work |
| JSON-LD and API schemas | Local references form recursive definition graphs |

## Inspect a compilation

```bash
oc-earley tier-report schema.json
oc-earley tier-report schema.json --format json
oc-earley compile schema.json
```

## Build

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --no-default-features
cargo test --all-features
uv run pytest -q
```

## Lineage and license

OC-Earley is a modified fork of `dottxt-ai/outlines-core`. The original Apache License 2.0 text is
retained in [LICENSE](LICENSE), and material modifications are summarized in [NOTICE](NOTICE).
