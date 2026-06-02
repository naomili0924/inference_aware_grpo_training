import torch

from vllm.v1.outputs import (
    ModelRunnerOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
)

from dataclasses import dataclass,field

@dataclass
class VLLMModelRunnerOutput(ModelRunnerOutput):
    num_accepted_spec_tokens: dict[str, int] = field(default_factory=dict)
    num_generated_tokens: dict[str, int] = field(default_factory=dict)
    
    