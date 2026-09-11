import ctypes
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def byte_vocabulary():
    return Vocabulary(256, {bytes([byte]): [byte] for byte in range(256)})


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def generate(compiled, data):
    guide = compiled.guide(max_rollback=32)
    words = (ctypes.c_uint32 * guide.mask_words)()
    for position, byte in enumerate(data):
        guide.write_mask_into(
            ctypes.addressof(words),
            len(words),
            ctypes.sizeof(words._type_),
        )
        assert words[byte // 32] & (1 << (byte % 32)), position
        guide.advance(byte, return_tokens=False)
    assert guide.is_accepting()
    guide.write_mask_into(
        ctypes.addressof(words),
        len(words),
        ctypes.sizeof(words._type_),
    )
    assert words[256 // 32] & (1 << (256 % 32))
    guide.advance(256, return_tokens=False)
    assert guide.is_finished()
    return data


def load_fixture(name):
    return json.loads((REGRESSIONS / name).read_text(encoding="utf-8"))


def recursive_object(depth):
    value = {"value": "é"}
    for _ in range(depth - 1):
        value = {"next": value, "value": "é"}
    return value


@pytest.mark.parametrize(
    ("fixture", "instance"),
    [
        ("recursive_object.json", recursive_object(12)),
        ("recursive_array.json", [[[[[[None]]]]]]),
        ("mutually_recursive.json", [[[[[[None]]]]]]),
        ("generic_recursive_json.json", [[None, True, 12, "é"]]),
        ("prefix_items.json", [1, 2, 2]),
        ("deep_acyclic_ref.json", "done"),
    ],
)
def test_guide_generated_values_pass_an_independent_validator(fixture, instance):
    schema = load_fixture(fixture)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    data = generate(
        CompiledSchema.from_json_schema(schema, byte_vocabulary()),
        canonical(instance),
    )

    decoded = json.loads(data.decode("utf-8"))
    assert decoded == instance
    assert validator.is_valid(decoded)


def test_invalid_recursive_value_is_rejected_by_both_engines():
    schema = load_fixture("recursive_object.json")
    validator = Draft202012Validator(schema)
    invalid = {"next": {"value": "x"}}
    data = canonical(invalid)
    compiled = CompiledSchema.from_json_schema(schema, byte_vocabulary())
    guide = compiled.guide()
    rejected = False

    for byte in data:
        try:
            guide.advance(byte, return_tokens=False)
        except ValueError:
            rejected = True
            break

    assert rejected
    assert not validator.is_valid(invalid)
