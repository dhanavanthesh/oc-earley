import ctypes

import pytest

from oc_earley import CompiledSchema, Vocabulary


RECURSIVE_NODE = {
    "$defs": {
        "node": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "next": {"$ref": "#/$defs/node"},
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    },
    "$ref": "#/$defs/node",
}


def mask_words(guide):
    words = (ctypes.c_uint32 * guide.mask_words)()
    guide.write_mask_into(
        ctypes.addressof(words),
        len(words),
        ctypes.sizeof(words._type_),
    )
    return list(words)


def enabled(words, token_id):
    return bool(words[token_id // 32] & (1 << (token_id % 32)))


def oracle(compiled, prefix, token_bytes, is_eos=False):
    recognizer = compiled.recognizer()
    if prefix:
        assert recognizer.advance(prefix) != "rejected"
    before = (recognizer.accepting, recognizer.live, recognizer.stats())
    checkpoint = recognizer.checkpoint()
    if is_eos:
        expected = recognizer.accepting
    else:
        expected = recognizer.advance(token_bytes) != "rejected" and recognizer.live
    recognizer.restore(checkpoint)
    assert (recognizer.accepting, recognizer.live, recognizer.stats()) == before
    return expected


def test_mask_matches_independent_trial_for_aliases_and_shared_prefixes():
    tokens = {
        b"{": [0],
        b'{"': [1],
        b'{"value"': [2],
        b'{"value":"x"}': [3, 19],
        b"null": [4],
        b"x": [5],
    }
    compiled = CompiledSchema.from_json_schema(
        RECURSIVE_NODE,
        Vocabulary(100, tokens),
    )
    guide = compiled.guide()

    assert compiled.backend in {"lalr", "earley"}
    assert guide.vocab_size == 101
    assert guide.mask_words == 4
    words = mask_words(guide)
    for token_bytes, token_ids in tokens.items():
        expected = oracle(compiled, b"", token_bytes)
        for token_id in token_ids:
            assert enabled(words, token_id) is expected
    assert enabled(words, 3) == enabled(words, 19)
    assert not enabled(words, 100)


def test_mask_crosses_multiple_terminals_and_commits_eos_explicitly():
    document = b'{"value":"x"}'
    compiled = CompiledSchema.from_json_schema(
        RECURSIVE_NODE,
        Vocabulary(100, {document: [3, 19], b"garbage": [7]}),
    )
    guide = compiled.guide()

    assert guide.get_tokens() == [3, 19]
    guide.advance(19, return_tokens=False)
    assert guide.is_accepting()
    assert not guide.is_finished()
    assert guide.get_tokens() == [100]
    guide.advance(100, return_tokens=False)
    assert guide.is_finished()
    assert guide.get_tokens() == []


def test_utf8_fragments_remain_live_across_token_boundaries():
    prefix = b'{"value":"'
    lead = b"\xc3"
    remainder = b'\xa9"}'
    compiled = CompiledSchema.from_json_schema(
        RECURSIVE_NODE,
        Vocabulary(100, {prefix: [0], lead: [1], remainder: [2]}),
    )
    guide = compiled.guide()

    assert guide.get_tokens() == [0]
    guide.advance(0, return_tokens=False)
    assert guide.get_tokens() == [1]
    guide.advance(1, return_tokens=False)
    assert guide.get_tokens() == [2]
    guide.advance(2, return_tokens=False)
    assert guide.is_accepting()
    guide.advance(100, return_tokens=False)
    assert guide.is_finished()


def test_mask_queries_do_not_change_parser_state_or_parser_counters():
    compiled = CompiledSchema.from_json_schema(
        RECURSIVE_NODE,
        Vocabulary(100, {b"{": [0], b'{"value":"x"}': [1]}),
    )
    guide = compiled.guide(max_rollback=8)
    before = (
        guide.state_fingerprint,
        guide.position,
        guide.is_accepting(),
        guide.get_allowed_rollback(),
        guide.parser_stats(),
    )
    expected = mask_words(guide)
    for _ in range(100):
        assert mask_words(guide) == expected
    after = (
        guide.state_fingerprint,
        guide.position,
        guide.is_accepting(),
        guide.get_allowed_rollback(),
        guide.parser_stats(),
    )

    assert after == before
    stats = guide.stats()
    assert stats["cache_misses"] == 1
    assert stats["cache_hits"] == 100


def test_conflicting_ids_and_empty_tokens_fail_before_mask_allocation():
    with pytest.raises(ValueError, match="token ID 7 maps to byte strings"):
        CompiledSchema.from_json_schema(
            RECURSIVE_NODE,
            Vocabulary(8, {b"a": [7], b"b": [7]}),
        )

    with pytest.raises(ValueError, match="must contain at least one byte"):
        CompiledSchema.from_json_schema(
            RECURSIVE_NODE,
            Vocabulary(1, {b"": [0]}),
        )
