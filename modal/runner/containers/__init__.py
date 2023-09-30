from .vllm_13b import Vllm13BContainer

# from .vllm_7b import Vllm7BContainer
# from .vllm_awq import VllmAWQ

from runner.shared.common import models_path
from typing import List


def _to_lower_list(l: List[str]):
    return [x.lower() for x in l]


vllm_13b_model_ids = [
    "PygmalionAI/mythalion-13b",
    "Gryphe/MythoMax-L2-13b",
    "Undi95/ReMM-SLERP-L2-13B",
    "meta-llama/Llama-2-13b-chat-hf",
    "NousResearch/Nous-Hermes-Llama2-13b",
    "mistralai/Mistral-7B-Instruct-v0.1",
]
_vllm_13b_models_lower = _to_lower_list(vllm_13b_model_ids)

# vllm_7b_model_ids = []
# _vllm_7b_models_lower = _to_lower_list(vllm_7b_model_ids)


# vllm_awq_model_ids = ["TheBloke/Xwin-LM-70B-V0.1-AWQ"]
# _vllm_awq_models_lower = _to_lower_list(vllm_awq_model_ids)

all_models = [
    *vllm_13b_model_ids,
    # *vllm_7b_model_ids,
    # *vllm_awq_model_ids,
]


def get_container(model: str):
    normalized_model_id = model.lower()
    model_path = models_path / normalized_model_id

    if model_path.exists():
        if normalized_model_id in _vllm_13b_models_lower:
            return Vllm13BContainer(str(model_path))
        # if normalized_model_id in _vllm_7b_models_lower:
        #     return Vllm7BContainer(str(model_path))
        # if normalized_model_id in _vllm_awq_models_lower:
        #     return VllmAWQ(str(model_path))

    raise ValueError(f"Invalid model: {model}")
