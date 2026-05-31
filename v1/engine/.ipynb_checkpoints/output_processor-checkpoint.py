from vllm.v1.engine import RequestState, OutputProcessor, OutputProcessorOutput
from inference_aware_grpo_training.output import VLLMCompletionOutput

from vllm.v1.engine import FinishReason, EngineCoreOutput
from vllm.sampling_params import RequestOutputKind

from vllm.outputs import (
    STREAM_FINISHED,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)

import numpy as np

from vllm.v1.metrics.stats import IterationStats

class VLLMRequestState(RequestState):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        spec_accept_rate = self.detokenizer.spec_accept_rate()

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids) :]

        # Concatenate routed experts on finish
        routed_experts = None
        if finished and self.routed_experts_chunks:
            routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)

        return VLLMCompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            routed_experts=routed_experts,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
            spec_accept_rate=spec_accept_rate,
        )

class VLLMOutputProcessor(OutputProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            num_valid_draft_token = engine_core_output.num_valid_draft_token
            num_generated_token = engine_core_output.num_generated_token
            
            if engine_core_output.routed_experts is not None:
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                req_state.is_prefilling = False

            # check if it is 
            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )

                spec_string = req_state.detokenizer.update_spec(num_valid_draft_token, num_generated_token)

                # cross frame the 
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
            ):
                if req_state.streaming_input:
                    request_output.finished = False

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                if req_state.streaming_input:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._finish_request(req_state)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

                    # Track per-request stats
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )
