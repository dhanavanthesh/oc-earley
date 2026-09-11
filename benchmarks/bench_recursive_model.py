import os


MODEL = os.environ.get("OC_EARLEY_BENCH_MODEL")
REVISION = os.environ.get("OC_EARLEY_BENCH_REVISION")


class RecursiveModelBenchmark:
    def setup(self):
        if not MODEL or not REVISION:
            raise NotImplementedError(
                "set OC_EARLEY_BENCH_MODEL and OC_EARLEY_BENCH_REVISION"
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            dtype=torch.float32,
        ).eval()
        self.inputs = self.tokenizer("Return JSON only:", return_tensors="pt")

    def time_one_forward_pass(self):
        with self.torch.inference_mode():
            self.model(**self.inputs)
