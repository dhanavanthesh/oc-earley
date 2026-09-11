import json
from pathlib import Path

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def vocabulary():
    return Vocabulary(1, {"unused": [0]})


def test_recursive_object_crosses_multiple_terminals_per_chunk():
    schema = json.loads((REGRESSIONS / "recursive_optional_property.json").read_text())
    compiled = CompiledSchema.from_json_schema(schema, vocabulary())
    recognizer = compiled.recognizer()

    assert recognizer.advance('{"next":{"value":"x"},"value":"x"}') == "accepting"
    assert recognizer.accepting
    assert recognizer.live
    assert recognizer.stats()


def test_rejected_chunk_does_not_change_the_recognizer():
    schema = json.loads((REGRESSIONS / "recursive_optional_property.json").read_text())
    recognizer = CompiledSchema.from_json_schema(schema, vocabulary()).recognizer()
    checkpoint = recognizer.checkpoint()

    assert recognizer.advance("invalid") == "rejected"
    recognizer.restore(checkpoint)
    assert recognizer.advance('{"value":"x"}') == "accepting"
