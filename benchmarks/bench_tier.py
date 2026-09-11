from pathlib import Path

from oc_earley import CompiledSchema, Vocabulary


REGRESSIONS = Path(__file__).parents[1] / "testdata" / "regressions"
SCHEMAS = {
    "finite_object": {
        "type": "object",
        "properties": {
            "active": {"type": "boolean"},
            "kind": {"enum": ["a", "b", "c"]},
        },
        "required": ["kind"],
        "additionalProperties": False,
    },
    "bounded_array": {
        "type": "array",
        "prefixItems": [{"const": 1}],
        "items": {"enum": [2, 3]},
        "minItems": 1,
        "maxItems": 8,
    },
    "deep_reference": (REGRESSIONS / "deep_acyclic_ref.json").read_text(),
}


class TierCompilerBenchmark:
    params = tuple(SCHEMAS)
    param_names = ["schema"]

    def setup(self, schema):
        self.schema = SCHEMAS[schema]
        self.vocabulary = Vocabulary.from_pretrained("gpt2")
        self.compiled = CompiledSchema.from_json_schema(self.schema, self.vocabulary)
        self.profile = self.compiled.profile()

    def time_compile(self, schema):
        CompiledSchema.from_json_schema(self.schema, self.vocabulary)

    def peakmem_compile(self, schema):
        return CompiledSchema.from_json_schema(self.schema, self.vocabulary)

    def track_schema_parse_ns(self, schema):
        return self.profile["schema_parse_ns"]

    def track_normalization_ns(self, schema):
        return self.profile["normalization_ns"]

    def track_grammar_lowering_ns(self, schema):
        return self.profile["grammar_lowering_ns"]

    def track_grammar_reduction_ns(self, schema):
        return self.profile["grammar_reduction_ns"]

    def track_scc_analysis_ns(self, schema):
        return self.profile["scc_analysis_ns"]

    def track_regular_certification_ns(self, schema):
        return self.profile["regular_certification_ns"]

    def track_nfa_construction_ns(self, schema):
        return self.profile["nfa_construction_ns"]

    def track_dfa_determinization_ns(self, schema):
        return self.profile["dfa_determinization_ns"]

    def track_vocabulary_projection_ns(self, schema):
        return self.profile["vocabulary_projection_ns"]

    def track_dfa_states(self, schema):
        return self.compiled.tier_report()["dfa_states"]

    def track_dfa_bytes(self, schema):
        return self.compiled.tier_report()["dfa_bytes"]
