import random

import pytest

from oc_earley import CompiledSchema, Vocabulary


RECURSIVE_ARRAY = {
    "$defs": {
        "node": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "prefixItems": [{"$ref": "#/$defs/node"}],
                    "items": False,
                },
            ]
        }
    },
    "$ref": "#/$defs/node",
}


def compile_array():
    return CompiledSchema.from_json_schema(
        RECURSIVE_ARRAY,
        Vocabulary(100, {b"[": [0], b"null": [1], b"]": [2], b"x": [3]}),
    )


def test_rejection_rollback_and_reset_match_fresh_replay():
    compiled = compile_array()
    guide = compiled.guide(max_rollback=8)
    initial = (guide.state_fingerprint, guide.get_tokens())

    with pytest.raises(ValueError, match="token ID 3 is not allowed"):
        guide.advance(3, return_tokens=False)
    assert (guide.state_fingerprint, guide.get_tokens()) == initial
    assert guide.get_allowed_rollback() == 0

    for token in [0, 0, 1, 2, 2]:
        guide.advance(token, return_tokens=False)
    assert guide.is_accepting()
    completed = (guide.state_fingerprint, guide.get_tokens())

    guide.rollback_state(2)
    for token in [2, 2]:
        guide.advance(token, return_tokens=False)
    assert (guide.state_fingerprint, guide.get_tokens()) == completed

    guide.reset()
    assert guide.get_allowed_rollback() == 0
    assert (guide.state_fingerprint, guide.get_tokens()) == initial


def test_zero_rollback_limit_does_not_retain_history():
    guide = compile_array().guide(max_rollback=0)
    guide.advance(1, return_tokens=False)
    assert guide.get_allowed_rollback() == 0
    with pytest.raises(ValueError, match="Cannot roll back"):
        guide.rollback_state(1)


def test_structural_state_is_opaque():
    guide = compile_array().guide()

    with pytest.raises(ValueError, match="state IDs are unavailable"):
        guide.get_state()
    assert guide.backend in {"lalr", "earley"}
    assert isinstance(guide.state_fingerprint, int)


def test_random_operations_match_fresh_replay():
    compiled = compile_array()
    guide = compiled.guide(max_rollback=8)
    committed = []
    randomizer = random.Random(0xEA41E7)

    for _ in range(500):
        allowed = guide.get_tokens()
        operation = randomizer.randrange(5)
        if guide.is_finished():
            operation = 3 if guide.get_allowed_rollback() else 4

        if operation == 0:
            assert guide.get_tokens() == allowed
        elif operation == 1 and allowed:
            token = randomizer.choice(allowed)
            guide.advance(token, return_tokens=False)
            committed.append(token)
        elif operation == 2 and 3 not in allowed:
            before = (
                guide.state_fingerprint,
                guide.get_allowed_rollback(),
                guide.get_tokens(),
            )
            with pytest.raises(ValueError):
                guide.advance(3, return_tokens=False)
            assert (
                guide.state_fingerprint,
                guide.get_allowed_rollback(),
                guide.get_tokens(),
            ) == before
        elif operation == 3 and guide.get_allowed_rollback():
            count = randomizer.randint(1, min(3, guide.get_allowed_rollback()))
            guide.rollback_state(count)
            del committed[-count:]
        elif operation == 4:
            guide.reset()
            committed.clear()

        replay = compiled.guide(max_rollback=8)
        for token in committed:
            replay.advance(token, return_tokens=False)
        assert (
            guide.state_fingerprint,
            guide.position,
            guide.is_accepting(),
            guide.is_finished(),
            guide.get_tokens(),
        ) == (
            replay.state_fingerprint,
            replay.position,
            replay.is_accepting(),
            replay.is_finished(),
            replay.get_tokens(),
        )
