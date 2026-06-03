from vllm.v1.engine.llm_engine import LLMEngine
from vllm.config import ParallelConfig, VllmConfig
from vllm.v1.executor.abstract import Executor
from vllm.usage.usage_lib import UsageContext
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
import vllm.v1.engine.core_client as _cc

from inference_aware_grpo_training.v1.engine.core_client import VLLMInprocClient


class VLLMEngine(LLMEngine):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None:
        # Temporarily patch make_client so super().__init__() creates VLLMInprocClient
        # instead of the default InprocClient, avoiding a second model load.
        original_make_client = _cc.EngineCoreClient.make_client

        def patched_make_client(
            multiprocess_mode,
            asyncio_mode,
            vllm_config,
            executor_class,
            log_stats,
        ):
            return VLLMInprocClient(vllm_config, executor_class, log_stats)

        _cc.EngineCoreClient.make_client = staticmethod(patched_make_client)
        try:
            super().__init__(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
                usage_context=usage_context,
                stat_loggers=stat_loggers,
                mm_registry=mm_registry,
                multiprocess_mode=multiprocess_mode,
            )
        finally:
            _cc.EngineCoreClient.make_client = staticmethod(original_make_client)
        # self.engine_core is now set to VLLMInprocClient by super().__init__()

    def get_spec_decode_stats(self) -> dict[str, dict]:
        """Return spec decode accept-rate stats for all completed requests."""
        return self.engine_core.engine_core.scheduler.get_spec_decode_stats()
