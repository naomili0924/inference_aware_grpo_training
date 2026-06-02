from vllm.v1.engine.llm_engine import LLMEngine
from vllm.config import ParallelConfig, VllmConfig
from vllm.v1.executor.abstract import Executor
from vllm.usage.usage_lib import UsageContext
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from inference_aware_grpo_training.v1.engine.core_client import make_client

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

        self.engine_core = make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )
        