from vllm import SamplingParams
from inference_aware_grpo_training import VLLM
import torch
import multiprocessing as mp


def main():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    llm = VLLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.15,
        max_model_len=1024,
        enforce_eager=False,

        # ❌ 不要用 speculative_config dict
        # ✔ vLLM 这里一般用 enable/disable 或 SpeculativeConfig对象
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    outputs = llm.generate(
        ["What is speculative decoding?"],
        sampling_params,
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()