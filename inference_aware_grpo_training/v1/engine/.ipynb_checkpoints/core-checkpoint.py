from vllm.v1.engine.core import EngineCore
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.config import ParallelConfig, VllmConfig

from inference_aware_grpo_training.v1.core.sched.scheduler import VLLMScheduler
from inference_aware_grpo_training.v1.outputs import VLLMModelRunnerOutput
from vllm.v1.executor import Executor
from collections.abc import Callable, Generator

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs


class VLLMEngineCore(EngineCore):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # IMPORTANT: pass required args to EngineCore if your vLLM version expects them
        super().__init__(vllm_config=vllm_config, executor_class=executor_class, log_stats=log_stats)

        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = VLLMScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )

    def compute_spec_decode_stats(
        self,
        scheduler_output: SchedulerOutput,
        model_output: VLLMModelRunnerOutput,
    ):
        """
        Returns:
            {
                req_id: {
                    "num_accepted_tokens": int,
                    "num_generated_tokens": int,
                    "accept_rate": float,
                }
            }
        """
        results = {}
        scheduled_spec_tokens = (
            scheduler_output.scheduled_spec_decode_tokens
        )
        
        for req_id in model_output.req_ids:
            # speculative draft tokens
            draft_tokens = scheduled_spec_tokens.get(req_id, [])
        
            # final sampled tokens from target model
            idx = model_output.req_id_to_index[req_id]
            generated_tokens = model_output.sampled_token_ids[idx]
        
            # count prefix match
            accepted = 0
            
            for d, g in zip(draft_tokens, generated_tokens):
                if d == g:
                    accepted += 1
                else:
                    break
        
            num_generated = len(generated_tokens)
            num_draft = len(draft_tokens)
        
            accept_rate = (
                accepted / num_draft
                if num_draft > 0
                else 0.0
            )
        
            results[req_id] = {
                "num_accepted_tokens": accepted,
                "num_generated_tokens": num_generated,
                "accept_rate": accept_rate,
            }
        return results

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
                tmp = self.compute_spec_decode_stats(scheduler_output, model_output)
                for key, value in tmp.items():
                    model_output.num_accepted_spec_tokens[key] = (
                        model_output.num_accepted_spec_tokens.get(key, 0)
                        + value['num_accepted_tokens']
                    )
                    
                    model_output.num_generated_tokens[key] = (
                        model_output.num_generated_tokens.get(key, 0)
                        + value['num_generated_tokens']
                    )

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0