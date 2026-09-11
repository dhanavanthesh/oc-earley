from oc_earley import CompiledSchema, Vocabulary

from .bench_structural_masks import SCHEMA


class StructuralGuideBenchmark:
    def setup(self):
        self.compiled = CompiledSchema.from_json_schema(
            SCHEMA,
            Vocabulary(3, {b"[": [0], b"null": [1], b"]": [2]}),
        )
        self.guide = self.compiled.guide(max_rollback=1)

    def time_advance_and_rollback(self):
        self.guide.advance(0, return_tokens=False)
        self.guide.rollback_state(1)

    def time_advance_and_reset(self):
        self.guide.advance(0, return_tokens=False)
        self.guide.reset()

    def track_serialized_guide_bytes(self):
        return len(self.guide.__reduce__()[1][0])
