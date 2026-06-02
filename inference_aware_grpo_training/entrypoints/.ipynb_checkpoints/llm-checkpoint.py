from vllm import LLM
from vllm.usage.usage_lib import UsageContext

from inference_aware_grpo_training.v1.engine.llm_engine import VLLMEngine


class VLLM(LLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # replace engine
        self.llm_engine = VLLMEngine.from_engine_args(
            engine_args=self.llm_engine.engine_args,
            usage_context=UsageContext.LLM_CLASS,
        )