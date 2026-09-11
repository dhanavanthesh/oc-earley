import json
from pathlib import Path

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def test_regular_and_structural_backends_are_selected_deterministically():
    vocabulary = Vocabulary(2, {"unused": [0], '"x"': [1]})
    recursive = json.loads((REGRESSIONS / "recursive_array.json").read_text())
    first = CompiledSchema.from_json_schema(recursive, vocabulary)
    second = CompiledSchema.from_json_schema(recursive, vocabulary)

    assert first.backend in {"lalr", "earley"}
    assert first.backend == second.backend
    assert first.tier_report() == second.tier_report()
    assert CompiledSchema.from_json_schema({"const": "x"}, vocabulary).backend == "whole_dfa"
