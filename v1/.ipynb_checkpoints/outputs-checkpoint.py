import torch

from vllm.v1.outputs import ModelRunnerOutput, KVConnectorOutput, ECConnectorOutput, LogprobsLists, LogprobsTensors,RoutedExpertsLists
from vllm.compilation.cuda_graph import CUDAGraphStat

from dataclasses import dataclass,field

@dataclass
class VLLMModelRunnerOutput(ModelRunnerOutput):
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None] | None = None

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None

    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    # Per-step routed experts data captured by the worker.
    # ``routing_data`` shape: (num_scheduled_tokens, num_layers,
    #                         num_experts_per_tok); expert IDs as uint8/uint16.
    # ``slot_mapping`` shape: (num_scheduled_tokens,); physical KV-cache
    #                         slot for each row of routing_data.
    # ``num_scheduled_tokens`` is step-level (total across all requests
    # in this step), not per-request. The scheduler persists this into
    # its slot buffer via ``slot_buffer[slot_mapping] = routing_data``.
    # ``None`` when ``enable_return_routed_experts`` is off.
    routed_experts: RoutedExpertsLists | None = None

    num_valid_draft_token_list: list[int] = field(default_factory=list)
    num_generate_token_list: list[int] = field(default_factory=list)
    
    