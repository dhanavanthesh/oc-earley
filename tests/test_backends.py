import json
import pickle
from pathlib import Path

import pytest

from oc_earley import CompiledSchema, Index, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def vocabulary(tokens, eos):
    return Vocabulary(eos, {token: [token_id] for token, token_id in tokens})


def test_regular_schema_produces_a_guide():
    schema = (REGRESSIONS / "deep_acyclic_ref.json").read_text()
    compiled = CompiledSchema.from_json_schema(
        schema,
        vocabulary([('"done"', 0), ('"other"', 1)], 2),
    )

    assert compiled.backend == "whole_dfa"
    assert compiled.tier_report()["reference_edges"] == 5
    guide = compiled.guide()
    assert 0 in guide.get_tokens()
    assert 1 not in guide.get_tokens()
    guide.advance(0)
    assert guide.is_finished()


def test_python_mapping_schema_is_supported():
    compiled = CompiledSchema.from_json_schema(
        {"allOf": [{"type": "string"}, {"const": "x"}]},
        vocabulary([('"x"', 0), ('"y"', 1)], 2),
    )

    assert compiled.backend == "whole_dfa"
    assert compiled.guide().get_tokens() == [0]


def test_recursive_schema_fails_before_decode():
    schema = json.loads(
        (REGRESSIONS / "recursive_optional_property.json").read_text()
    )
    compiled = CompiledSchema.from_json_schema(schema, vocabulary([("null", 0)], 1))

    assert compiled.backend == "structural_pending"
    assert compiled.tier_report()["diagnostics"]
    with pytest.raises(ValueError, match="structural backend is required"):
        compiled.guide()


def test_compiled_schema_pickle_recompiles_the_same_contract():
    compiled = CompiledSchema.from_json_schema(
        {"const": "done"},
        vocabulary([('"done"', 0)], 1),
    )
    restored = pickle.loads(pickle.dumps(compiled))

    assert restored == compiled
    assert restored.tier_report() == compiled.tier_report()
    assert restored.guide().get_tokens() == [0]


def test_serialized_headers_reject_invalid_inputs():
    compiled = CompiledSchema.from_json_schema(
        {"const": "done"},
        vocabulary([('"done"', 0)], 1),
    )
    binary = compiled.__reduce__()[1][0]

    with pytest.raises(ValueError, match="truncated header"):
        CompiledSchema.from_binary(binary[:4])
    with pytest.raises(ValueError, match="wrong object kind"):
        Index.from_binary(binary)

    wrong_version = list(binary)
    wrong_version[8] = 2
    with pytest.raises(ValueError, match="format version 2 is unsupported"):
        CompiledSchema.from_binary(wrong_version)

    with pytest.raises(ValueError, match="invalid magic bytes"):
        CompiledSchema.from_binary([0] * 13)
