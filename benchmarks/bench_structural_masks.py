import ctypes

from oc_earley import CompiledSchema, Vocabulary


SCHEMA = {
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


def vocabulary():
    tokens = {bytes([byte]): [byte] for byte in range(256)}
    tokens[b"[[null]]"] = [300, 301]
    return Vocabulary(512, tokens)


def fill(guide, mask):
    guide.write_mask_into(
        ctypes.addressof(mask),
        len(mask),
        ctypes.sizeof(mask._type_),
    )


class StructuralMaskBenchmark:
    def setup(self):
        self.compiled = CompiledSchema.from_json_schema(SCHEMA, vocabulary())
        self.guide = self.compiled.guide()
        self.mask = (ctypes.c_uint32 * self.guide.mask_words)()
        fill(self.guide, self.mask)

    def time_warm_mask(self):
        fill(self.guide, self.mask)

    def time_cold_mask(self):
        guide = self.compiled.guide()
        mask = (ctypes.c_uint32 * guide.mask_words)()
        fill(guide, mask)

    def track_trie_edges_visited(self):
        return self.guide.stats()["trie_edges_visited"]

    def track_cache_bytes(self):
        return self.guide.stats()["cache_bytes"]
