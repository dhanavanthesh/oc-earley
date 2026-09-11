import json

import pytest
from jsonschema import Draft202012Validator

from oc_earley import CompiledSchema, Vocabulary


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compile_candidates(schema, candidates):
    encoded = [canonical_json(candidate) for candidate in candidates]
    assert len(encoded) == len(set(encoded))
    vocabulary = Vocabulary(
        len(encoded),
        {token: [token_id] for token_id, token in enumerate(encoded)},
    )
    compiled = CompiledSchema.from_json_schema(schema, vocabulary)
    assert compiled.backend == "whole_dfa"
    return compiled


def dfa_accepts(compiled, token_id):
    guide = compiled.guide()
    try:
        guide.advance(token_id, False)
    except ValueError:
        return False
    return guide.is_finished()


@pytest.mark.parametrize(
    ("schema", "candidates"),
    [
        (
            {"enum": [None, True, "x", 2]},
            [None, False, True, "", "x", 1, 2, [], {}],
        ),
        (
            {"allOf": [{"type": "string"}, {"const": "x"}]},
            [None, "", "x", "xx", 1],
        ),
        (
            {
                "type": "array",
                "prefixItems": [{"const": 1}],
                "items": {"const": 2},
                "minItems": 1,
                "maxItems": 3,
            },
            [[], [1], [2], [1, 2], [1, 2, 2], [1, 3], [1, 2, 2, 2]],
        ),
        (
            {
                "type": "object",
                "properties": {
                    "a": {"const": 1},
                    "b": {"enum": [True, False]},
                },
                "required": ["a"],
                "additionalProperties": False,
            },
            [
                {},
                {"a": 1},
                {"a": 2},
                {"a": 1, "b": True},
                {"b": False, "a": 1},
                {"a": 1, "c": None},
            ],
        ),
        (
            {"anyOf": [{"const": 1}, {"const": "one"}]},
            [None, 0, 1, 10, "1", "one"],
        ),
    ],
)
def test_whole_dfa_matches_independent_draft_2020_12_validator(schema, candidates):
    validator = Draft202012Validator(schema)
    compiled = compile_candidates(schema, candidates)

    for token_id, candidate in enumerate(candidates):
        expected = validator.is_valid(candidate)
        actual = dfa_accepts(compiled, token_id)
        assert actual == expected, canonical_json(candidate)


def test_token_mask_membership_matches_live_dfa_transitions():
    schema = {"const": 10}
    vocabulary = Vocabulary(
        5,
        {
            "1": [0],
            "10": [1],
            "100": [2],
            '"10"': [3],
            "0": [4],
        },
    )
    compiled = CompiledSchema.from_json_schema(schema, vocabulary)
    guide = compiled.guide()

    assert guide.get_tokens() == [0, 1]
    guide.advance(0, False)
    assert not guide.is_finished()
    assert guide.get_tokens() == [4]
    guide.advance(4, False)
    assert guide.is_finished()

    guide = compiled.guide()
    assert guide.accepts_tokens([1])
    assert not guide.accepts_tokens([2])
    assert not guide.accepts_tokens([3])


def test_uncertified_unicode_length_automaton_is_explicitly_rejected():
    vocabulary = Vocabulary(1, {'"a"': [0]})

    with pytest.raises(ValueError, match="Unicode length constraints"):
        CompiledSchema.from_json_schema(
            {"type": "string", "minLength": 1, "maxLength": 2},
            vocabulary,
        )
