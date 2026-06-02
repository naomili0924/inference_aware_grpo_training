from setuptools import setup, find_packages

setup(
    name="inference_aware_grpo_training",
    version="0.1.0",
    packages=find_packages(),

    python_requires=">=3.10",

    install_requires=[
        # ===== MUST match vLLM requirement =====
        "torch==2.10.0",

        # ===== vLLM stable 2026 range =====
        "vllm==0.19.0",

        # ===== core ML stack =====
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "datasets>=2.19.0",

        # ===== utilities =====
        "numpy",
        "tqdm",
        "pydantic",
        "sentencepiece",
        "protobuf",
        "tiktoken",
    ],

    include_package_data=True,
    zip_safe=False,
)