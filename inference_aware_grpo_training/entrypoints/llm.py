from typing import Any, Optional

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.entrypoints.chat_utils import load_chat_template

from inference_aware_grpo_training.v1.engine.llm_engine import VLLMEngine


class VLLM(LLM):
    """
    Safe wrapper:
    - keeps inheritance from LLM (API compatibility)
    - avoids LLM.__init__ engine creation
    - uses ONLY custom VLLMEngine
    """

    def __init__(self, *args, **kwargs):
        # ----------------------------
        # 1. save raw args (optional)
        # ----------------------------
        self._init_args = args
        self._init_kwargs = kwargs

        # ----------------------------
        # 2. build engine args directly
        # ----------------------------
        engine_args = EngineArgs(*args, **kwargs)

        # ----------------------------
        # 3. ONLY ONE engine init
        # ----------------------------
        self.llm_engine = VLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS,
        )

        # ----------------------------
        # 4. emulate LLM base fields
        # ----------------------------
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Optional[dict[str, Any]] = None

        # tasks / runtime metadata
        self.supported_tasks = self.llm_engine.get_supported_tasks()
        self.pooling_task = self.model_config.get_pooling_task(self.supported_tasks)

        self.runner_type = self.model_config.runner_type
        self.renderer = self.llm_engine.renderer
        self.io_processor = self.llm_engine.io_processor
        self.input_processor = self.llm_engine.input_processor

        # ----------------------------
        # 5. chat template handling
        # ----------------------------
        chat_template = kwargs.get("chat_template", None)
        self.chat_template = load_chat_template(chat_template)

        # (optional, if your repo uses it)
        self.chat_template_config = ChatTemplateConfig(
            chat_template=self.chat_template
        )

        # ----------------------------
        # 6. pooling processors
        # ----------------------------
        self.pooling_io_processors = init_pooling_io_processors(
            supported_tasks=self.supported_tasks,
            model_config=self.model_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )

        # cache repr
        self._cached_repr: str | None = None