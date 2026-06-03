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
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 5,
        },
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducible accept rates
        max_tokens=128,
    )

    # Prompt-lookup (ngram) spec decode proposes candidate tokens by finding
    # the last few generated tokens as a substring of the INPUT prompt.
    # It works best when the model quotes or paraphrases the input.
    prompts = [
        "Summarize: Speculative decoding uses a fast draft model to propose tokens.",
        "Summarize: GRPO is a RL algorithm for training language models.",
    ]

    outputs = llm.generate(prompts, sampling_params)

    # get_spec_decode_stats() returns {request_id: {num_accepted, num_draft, accept_rate}}
    # Only requests where spec decode was triggered appear in the dict.
    spec_stats = llm.get_spec_decode_stats()

    print("=== Spec Decode Accept Rates ===\n")
    for output in outputs:
        req_id = output.request_id
        text = output.outputs[0].text
        stats = spec_stats.get(req_id)
        if stats and stats["num_draft"] > 0:
            print(
                f"[{req_id}]  accept_rate={stats['accept_rate']:.1%}  "
                f"({stats['num_accepted']}/{stats['num_draft']} draft tokens accepted)"
            )
        else:
            print(f"[{req_id}]  spec decode not triggered for this request")
        print(f"  Output: {text.strip()[:100]}")
        print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
