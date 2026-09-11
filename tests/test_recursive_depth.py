import json
from pathlib import Path

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def test_recursive_object_reaches_depth_one_hundred():
    schema = json.loads((REGRESSIONS / "recursive_object.json").read_text())
    compiled = CompiledSchema.from_json_schema(schema, Vocabulary(1, {"unused": [0]}))
    value = '{"value":"x"}'
    for _ in range(99):
        value = f'{{"next":{value},"value":"x"}}'

    recognizer = compiled.recognizer()
    assert recognizer.advance(value) == "accepting"
    assert recognizer.accepting


def test_mutual_recursion_reaches_depth_one_hundred():
    schema = json.loads((REGRESSIONS / "mutually_recursive.json").read_text())
    compiled = CompiledSchema.from_json_schema(schema, Vocabulary(1, {"unused": [0]}))
    value = "[" * 100 + "null" + "]" * 100

    assert compiled.recognizer().advance(value) == "accepting"
    assert compiled.recognizer().advance("[true]") == "rejected"


def test_generic_recursive_arrays_preserve_primitive_branches():
    schema = json.loads((REGRESSIONS / "generic_recursive_json.json").read_text())
    compiled = CompiledSchema.from_json_schema(schema, Vocabulary(1, {"unused": [0]}))

    for value in ("null", "true", "12", '"x"', '[[null,true,12,"x"]]'):
        assert compiled.recognizer().advance(value) == "accepting"
    assert compiled.recognizer().advance("{}") == "rejected"


def test_recursive_array_reaches_depth_one_hundred():
    schema = json.loads((REGRESSIONS / "recursive_array.json").read_text())
    compiled = CompiledSchema.from_json_schema(schema, Vocabulary(1, {"unused": [0]}))
    value = "[" * 100 + "null" + "]" * 100

    recognizer = compiled.recognizer()
    assert recognizer.advance(value) == "accepting"
