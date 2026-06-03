from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor import Executor
from vllm.config import VllmConfig

from inference_aware_grpo_training.v1.engine.core import VLLMEngineCore


class VLLMInprocClient(InprocClient):
    """
    Proper vLLM v1 in-process client wrapper.
    DO NOT manually assign engine_core.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        # Skip super().__init__() — it would create a plain EngineCore (a full
        # model load) that we immediately discard. Create VLLMEngineCore directly.
        self.engine_core = VLLMEngineCore(
            vllm_config,
            executor_class,
            log_stats,
        )


def make_client(
    multiprocess_mode: bool,
    asyncio_mode: bool,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
):
    return VLLMInprocClient(
        vllm_config,
        executor_class,
        log_stats,
    )