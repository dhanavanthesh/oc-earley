import ctypes
import json
import pickle
from pathlib import Path

from oc_earley import CompiledSchema, Guide, Index, Recognizer, Vocabulary


FIXTURES = Path(__file__).parents[1] / "testdata" / "regressions"


def exercise(schema, document, expected_backend):
    vocabulary = Vocabulary(2, {document: [0], b"invalid": [1]})
    compiled = CompiledSchema.from_json_schema(schema, vocabulary)
    guide = compiled.guide(max_rollback=4)
    mask = (ctypes.c_uint32 * guide.mask_words)()

    assert compiled.backend == expected_backend
    assert isinstance(compiled.recognizer(), Recognizer)
    guide.write_mask_into(
        ctypes.addressof(mask),
        len(mask),
        ctypes.sizeof(mask._type_),
    )
    assert mask[0] & 1
    guide.advance(0, return_tokens=False)
    assert guide.is_accepting()
    guide.rollback_state(1)
    assert guide.get_tokens() == [0]
    guide.advance(0, return_tokens=False)
    guide = pickle.loads(pickle.dumps(guide))
    assert guide.get_tokens() == [2]
    guide.advance(2, return_tokens=False)
    assert guide.is_finished()
    guide.reset()
    assert guide.get_tokens() == [0]


def main():
    vocabulary = Vocabulary(1, {b"a": [0]})
    legacy = Guide(Index("a", vocabulary))
    assert legacy.get_tokens() == [0]

    exercise({"const": "x"}, b'"x"', "whole_dfa")
    exercise(
        json.loads((FIXTURES / "recursive_array.json").read_text()),
        b"[[null]]",
        "lalr",
    )
    exercise(
        json.loads((FIXTURES / "ambiguous_anyof.json").read_text()),
        b"[[null]]",
        "earley",
    )


if __name__ == "__main__":
    main()
