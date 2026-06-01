from vllm.v1.engine.detokenizer import IncrementalDetokenizer

class VLLMIncrementalDetokenizer(IncrementalDetokenizer):

    def __init__(self):
        super().__init__()
        self.num_total_accepted_spec_tokens = 0
        self.num_total_generated_tokens = 0

    def update_spec(self, new_accepted_spec_tokens: int, new_generated_tokens: int) -> str | None:
        self.num_total_accepted_spec_tokens += new_accepted_spec_tokens
        self.num_total_generated_tokens += new_generated_tokens
        return None

    def spec_accept_rate(self) -> float | None:
        if self.num_total_generated_tokens == 0:
            return None
        return float(self.num_total_accepted_spec_tokens) / self.num_total_generated_tokens
