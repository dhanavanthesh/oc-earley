import json
from pathlib import Path

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def test_checkpoint_restore_and_replay_are_equivalent():
    schema = json.loads((REGRESSIONS / "recursive_optional_property.json").read_text())
    recognizer = CompiledSchema.from_json_schema(
        schema,
        Vocabulary(1, {"unused": [0]}),
    ).recognizer()
    assert recognizer.advance('{"next":') == "live"
    checkpoint = recognizer.checkpoint()
    tail = '{"value":"x"},"value":"x"}'

    assert recognizer.advance(tail) == "accepting"
    first_stats = recognizer.stats()
    recognizer.restore(checkpoint)
    assert recognizer.advance(tail) == "accepting"
    assert recognizer.stats() == first_stats
