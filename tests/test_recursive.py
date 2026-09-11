import json
from pathlib import Path

import pytest

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"


def compile_fixture(name):
    schema = (REGRESSIONS / name).read_text()
    return CompiledSchema.from_json_schema(schema, Vocabulary(1, {"null": [0]}))


@pytest.mark.parametrize(
    "name",
    ["recursive_optional_property.json", "recursive_required_property.json"],
)
def test_recursive_property_is_preserved_in_the_structural_plan(name):
    compiled = compile_fixture(name)
    report = compiled.tier_report()
    pointers = {
        pointer
        for component in report["sccs"]
        for pointer in component["source_pointers"]
    }

    assert compiled.backend == "structural_pending"
    assert report["reference_edges"] == 2
    assert any(pointer.endswith("/properties/next") for pointer in pointers)
    assert any(
        component["failure_reason"] == "RecursiveSymbolInInterior"
        for component in report["sccs"]
    )
    with pytest.raises(ValueError, match="structural backend is required"):
        compiled.guide()


def test_recursive_report_is_byte_stable():
    first = compile_fixture("recursive_required_property.json").tier_report()
    second = compile_fixture("recursive_required_property.json").tier_report()

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second,
        sort_keys=True,
        separators=(",", ":"),
    )
