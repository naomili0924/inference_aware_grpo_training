from vllm.v1.engine.core_client import InprocClient
from inference_aware_grpo_training.v1.engine.core import VLLMEngineCore

from vllm.v1.executor import Executor
from vllm.config import VllmConfig

def make_client(
    multiprocess_mode: bool,
    asyncio_mode: bool,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
) -> "EngineCoreClient":
    return VLLMInprocClient(vllm_config, executor_class, log_stats)
    

class VLLMInprocClient(InprocClient):

    def __init__(self, *args, **kwargs):
        self.engine_core = VLLMEngineCore(*args, **kwargs)