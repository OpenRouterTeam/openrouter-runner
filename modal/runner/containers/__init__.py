# from .lorean_7b import Lorean7BContainer

# from .vllm_7b import Vllm7BContainer
# from .vllm_awq import VllmAWQ

from .vllm_a100_128k import Vllm_A100_128K_Container
from .vllm_a100_32k import Vllm_A100_32K_Container
from .vllm_mid import VllmMidContainer

from shared.volumes import get_model_path
from typing import List


def _to_lower_list(l: List[str]):
    return [x.lower() for x in l]


vllm_mid_model_ids = [
    "PygmalionAI/mythalion-13b",
    "Gryphe/MythoMax-L2-13b",
    "Undi95/ReMM-SLERP-L2-13B",
    "meta-llama/Llama-2-13b-chat-hf",
    "NousResearch/Nous-Hermes-Llama2-13b",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
]
_vllm_mid_models_lower = _to_lower_list(vllm_mid_model_ids)


vllm_a100_128k_model_ids = [
    "NousResearch/Yarn-Mistral-7b-128k",
]
_vllm_a100_128k_model_lower = _to_lower_list(vllm_a100_128k_model_ids)

vllm_a100_32k_model_ids = [
    "NousResearch/Nous-Capybara-34B",
]
_vllm_a100_32k_models_lower = _to_lower_list(vllm_a100_32k_model_ids)


# vllm_awq_model_ids = ["TheBloke/Xwin-LM-70B-V0.1-AWQ"]
# _vllm_awq_models_lower = _to_lower_list(vllm_awq_model_ids)

all_models = [
    *vllm_mid_model_ids,
    *vllm_a100_32k_model_ids,
    *vllm_a100_128k_model_ids,
    # *vllm_7b_model_ids,
    # *vllm_awq_model_ids,
]


def get_container(model: str):
    normalized_model_id = model.lower()
    model_path = get_model_path(normalized_model_id)

    # if normalized_model_id == "lorean":
    #     return Lorean7BContainer(str(model_path))

    if model_path.exists():
        if normalized_model_id in _vllm_a100_32k_models_lower:
            return Vllm_A100_32K_Container(str(model_path))

        if normalized_model_id in _vllm_a100_128k_model_lower:
            return Vllm_A100_128K_Container(str(model_path))

        if normalized_model_id in _vllm_mid_models_lower:
            return VllmMidContainer(str(model_path))

        # if normalized_model_id in _vllm_7b_models_lower:
        #     return Vllm7BContainer(str(model_path))
        # if normalized_model_id in _vllm_awq_models_lower:
        #     return VllmAWQ(str(model_path))

    raise ValueError(f"Invalid model: {model}")
