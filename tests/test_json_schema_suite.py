import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from oc_earley import CompiledSchema, Vocabulary


SUITE = Path(__file__).parents[1] / "testdata" / "json-schema-test-suite"
CASES = json.loads((SUITE / "cases.json").read_text(encoding="utf-8"))


def normalize_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_numbers(item) for key, item in value.items()}
    return value


def canonical_json(value):
    return json.dumps(
        normalize_numbers(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def vocabulary_for(tests):
    tokens = {}
    for token_id, test in enumerate(tests):
        tokens.setdefault(canonical_json(test["data"]), []).append(token_id)
    next_id = len(tests)
    for byte in range(256):
        tokens.setdefault(bytes([byte]), []).append(next_id)
        next_id += 1
    return Vocabulary(next_id, tokens)


@pytest.mark.parametrize(
    "record",
    CASES,
    ids=lambda record: f"{record['source']}:{record['index']}",
)
def test_pinned_draft_2020_12_group(record):
    group = record["group"]
    validator = Draft202012Validator(group["schema"])
    compiled = CompiledSchema.from_json_schema(
        group["schema"],
        vocabulary_for(group["tests"]),
    )

    assert compiled.backend == "whole_dfa"
    for token_id, test in enumerate(group["tests"]):
        expected = validator.is_valid(test["data"])
        assert expected == test["valid"]
        guide = compiled.guide()
        try:
            guide.advance(token_id, False)
            actual = guide.is_finished()
        except ValueError:
            actual = False
        assert actual == expected, test["description"]
