from shared.volumes import get_model_path

from .vllm_unified import (
    VllmContainer_7B,
    VllmContainerA100_40G,
    VllmContainerA100_80G,
    VllmContainerA100_160G,
    VllmContainerA100_160G_Isolated,
)


def _to_lower_list(l: list[str]):
    return [x.lower() for x in l]


vllm_7b_model_ids = [
    "Intel/neural-chat-7b-v3-1",
]
_vllm_7b_models_lower = _to_lower_list(vllm_7b_model_ids)

vllm_mid_model_ids = [
    "PygmalionAI/mythalion-13b",
    "jebcarter/Psyfighter-13B",
    "KoboldAI/LLaMA2-13B-Psyfighter2",
    "Austism/chronos-hermes-13b-v2",
]
_vllm_mid_models_lower = _to_lower_list(vllm_mid_model_ids)

vllm_top_model_ids = [
    "NeverSleep/Noromaid-20b-v0.1.1",
]
_vllm_top_model_lower = _to_lower_list(vllm_top_model_ids)

vllm_a100_80gb_32k_model_ids = []
_vllm_a100_80gb_32k_models_lower = _to_lower_list(vllm_a100_80gb_32k_model_ids)

vllm_a100_160gb_8k_models = [
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    "NeverSleep/Noromaid-v0.1-mixtral-8x7b-Instruct-v3",
]
_vllm_a100_160gb_8k_models_lower = _to_lower_list(vllm_a100_160gb_8k_models)

vllm_a100_160gb_isolated_models = [
    "jondurbin/bagel-34b-v0.2",
]
_vllm_a100_160gb_isolated_models_lower = _to_lower_list(
    vllm_a100_160gb_isolated_models
)


all_models = [
    *vllm_7b_model_ids,
    *vllm_mid_model_ids,
    *vllm_top_model_ids,
    *vllm_a100_80gb_32k_model_ids,
    *vllm_a100_160gb_8k_models,
    *vllm_a100_160gb_isolated_models,
]
all_models_lower = _to_lower_list(all_models)


def get_container(model: str):
    normalized_model_id = model.lower()
    model_path = get_model_path(normalized_model_id)

    # if normalized_model_id == "lorean":
    #     return Lorean7BContainer(str(model_path))

    if model_path.exists():
        if normalized_model_id in _vllm_7b_models_lower:
            return VllmContainer_7B(str(model_path))

        if normalized_model_id in _vllm_a100_160gb_8k_models_lower:
            return VllmContainerA100_160G(str(model_path))

        if normalized_model_id in _vllm_a100_160gb_isolated_models_lower:
            return VllmContainerA100_160G_Isolated(str(model_path))

        if normalized_model_id in _vllm_a100_80gb_32k_models_lower:
            return VllmContainerA100_80G(str(model_path), max_model_len=32_000)

        if normalized_model_id in _vllm_mid_models_lower:
            return VllmContainerA100_40G(str(model_path))

        if normalized_model_id in _vllm_top_model_lower:
            return VllmContainerA100_80G(str(model_path))

        # if normalized_model_id in _vllm_awq_models_lower:
        #     return VllmAWQ(str(model_path))

    raise ValueError(f"Invalid model: {model}")
