import ctypes

import pytest

from oc_earley import CompiledSchema, Guide, Index, Vocabulary


def make_guide():
    vocabulary = Vocabulary(
        70,
        {
            "a": [0],
            "b": [31],
            "c": [32],
            "d": [63],
        },
    )
    return Guide(Index("a|b|c|d", vocabulary))


def test_packed_mask_layout_is_lsb_first_u32():
    guide = make_guide()
    mask = (ctypes.c_uint32 * 3)(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

    guide.write_mask_into(ctypes.addressof(mask), len(mask), ctypes.sizeof(mask._type_))

    assert list(mask) == [0x80000001, 0x80000001, 0]


def test_packed_mask_rejects_address_range_overflow():
    guide = make_guide()
    mask = (ctypes.c_uint32 * 3)()

    with pytest.raises(ValueError, match="byte length overflowed"):
        guide.write_mask_into(
            ctypes.addressof(mask),
            1 << (ctypes.sizeof(ctypes.c_void_p) * 8 - 1),
            ctypes.sizeof(mask._type_),
        )


def test_every_binary_object_uses_the_versioned_header():
    vocabulary = Vocabulary(3, {"a": [0], '"a"': [2]})
    index = Index("a", vocabulary)
    guide = Guide(index)
    compiled = CompiledSchema.from_json_schema({"const": "a"}, vocabulary)

    for value in (vocabulary, index, guide, compiled):
        binary = value.__reduce__()[1][0]
        assert binary[:8] == b"OCEARLEY"
        assert binary[8:12] == bytes([3, 0, 0, 0])


def test_binary_decoders_reject_trailing_and_legacy_payloads():
    vocabulary = Vocabulary(1, {"a": [0]})
    binary = vocabulary.__reduce__()[1][0]

    with pytest.raises(ValueError, match="trailing data"):
        Vocabulary.from_binary(binary + bytes([0]))
    with pytest.raises(ValueError, match="invalid magic bytes"):
        Vocabulary.from_binary([1] * 13)
