from vllm.v1.engine import EngineCoreOutput

class VLLMEngineCoreOutput(EngineCoreOutput):
    
    # For speculative decoding
    num_valid_draft_token: int = 0
    num_generated_token: int = 0
    
class VLLMEngineCoreOutputs(EngineCoreOutputs):
    outputs = list(VLLMEngineCoreOutput)
