import pickle

import pytest

from oc_earley import CompiledSchema, Guide, Vocabulary


SCHEMA = {
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


def test_structural_pickle_recompiles_and_replays_the_committed_prefix():
    compiled = CompiledSchema.from_json_schema(
        SCHEMA,
        Vocabulary(100, {b"[": [0], b"null": [1], b"]": [2]}),
    )
    guide = compiled.guide(max_rollback=4)
    for token in [0, 0, 1]:
        guide.advance(token, return_tokens=False)

    restored = pickle.loads(pickle.dumps(guide))

    assert restored == guide
    assert restored.backend == guide.backend
    assert restored.get_tokens() == guide.get_tokens()
    assert restored.get_allowed_rollback() == guide.get_allowed_rollback()
    for token in [2, 2, 100]:
        restored.advance(token, return_tokens=False)
    assert restored.is_finished()


def test_finished_structural_pickle_remains_finished():
    compiled = CompiledSchema.from_json_schema(
        SCHEMA,
        Vocabulary(100, {b"null": [1]}),
    )
    guide = compiled.guide()
    guide.advance(1, return_tokens=False)
    guide.advance(100, return_tokens=False)

    restored = pickle.loads(pickle.dumps(guide))

    assert restored.is_finished()
    assert restored.get_tokens() == []


def test_structural_pickle_rejects_trailing_and_wrong_version_data():
    compiled = CompiledSchema.from_json_schema(
        SCHEMA,
        Vocabulary(100, {b"null": [1]}),
    )
    binary = compiled.guide().__reduce__()[1][0]

    with pytest.raises(ValueError, match="trailing data"):
        Guide.from_binary(binary + b"\x00")
    wrong_version = bytearray(binary)
    wrong_version[8] = 3
    with pytest.raises(ValueError, match="format version 3 is unsupported"):
        Guide.from_binary(wrong_version)
