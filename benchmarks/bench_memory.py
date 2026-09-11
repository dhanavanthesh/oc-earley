from oc_earley import CompiledSchema, Vocabulary

from .bench_structural_masks import SCHEMA, vocabulary


class StructuralMemoryBenchmark:
    def setup(self):
        self.vocabulary = vocabulary()

    def peakmem_compile(self):
        return CompiledSchema.from_json_schema(SCHEMA, self.vocabulary)

    def peakmem_one_hundred_guides(self):
        compiled = CompiledSchema.from_json_schema(SCHEMA, self.vocabulary)
        return [compiled.guide() for _ in range(100)]

    def track_compiled_payload_bytes(self):
        compiled = CompiledSchema.from_json_schema(SCHEMA, self.vocabulary)
        return len(compiled.__reduce__()[1][0])
