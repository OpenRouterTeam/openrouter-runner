from .vllm_13b import Vllm13BContainer
from ..shared.volume import models_path
from typing import List


def _to_lower_list(l: List[str]):
    return [x.lower() for x in l]


vllm_13b_model_ids = [
    "PygmalionAI/mythalion-13b",
    # "Gryphe/MythoMax-L2-13b",
    # "Undi95/ReMM-SLERP-L2-13B",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "NousResearch/Nous-Hermes-Llama2-13b",
]
_vllm_13b_models_lower = _to_lower_list(vllm_13b_model_ids)


def get_container(model: str):
    normalized_model_id = model.lower()
    model_path = models_path / normalized_model_id

    if model_path.exists():
        if normalized_model_id in _vllm_13b_models_lower:
            return Vllm13BContainer(model_path, 4096)

    raise ValueError(f"Invalid model: {model}")
