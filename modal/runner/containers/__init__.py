from shared.volumes import get_model_path

from .vllm_unified import (
    VllmContainer_7B,
    VllmContainerA100_40G,
    VllmContainerA100_80G,
    VllmContainerA100_160G,
)


def _to_lower_list(l: list[str]):
    return [x.lower() for x in l]

vllm_7b_model_ids = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "Intel/neural-chat-7b-v3-1",
    "Undi95/Toppy-M-7B",
]
_vllm_7b_models_lower = _to_lower_list(vllm_7b_model_ids)

vllm_mid_model_ids = [
    "PygmalionAI/mythalion-13b",
    "Gryphe/MythoMax-L2-13b",
    "Undi95/ReMM-SLERP-L2-13B",
    "meta-llama/Llama-2-13b-chat-hf",
    "NousResearch/Nous-Hermes-Llama2-13b",
    "microsoft/Orca-2-13b",
    "jebcarter/Psyfighter-13B",
    "KoboldAI/LLaMA2-13B-Psyfighter2",
]
_vllm_mid_models_lower = _to_lower_list(vllm_mid_model_ids)

vllm_top_model_ids = [
    "NeverSleep/Noromaid-20b-v0.1.1",
]
_vllm_top_model_lower = _to_lower_list(vllm_top_model_ids)

vllm_a100_80gb_128k_model_ids = [
    "NousResearch/Yarn-Mistral-7b-128k",
]
_vllm_a100_80gb_128k_model_lower = _to_lower_list(vllm_a100_80gb_128k_model_ids)

vllm_a100_80gb_32k_model_ids = [
    "NousResearch/Nous-Capybara-34B",
]
_vllm_a100_80gb_32k_models_lower = _to_lower_list(vllm_a100_80gb_32k_model_ids)

vllm_a100_160gb_16k_models = [
    "ehartford/dolphin-2.5-mixtral-8x7b",
    "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
]
_vllm_a100_160gb_16k_models_lower = _to_lower_list(vllm_a100_160gb_16k_models)


all_models = [
    *vllm_7b_model_ids,
    *vllm_mid_model_ids,
    *vllm_top_model_ids,
    *vllm_a100_80gb_32k_model_ids,
    *vllm_a100_80gb_128k_model_ids,
    *vllm_a100_160gb_16k_models
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

        if normalized_model_id in _vllm_a100_160gb_16k_models_lower:
            return VllmContainerA100_160G(str(model_path), max_model_len=16_000)

        if normalized_model_id in _vllm_a100_80gb_32k_models_lower:
            return VllmContainerA100_80G(str(model_path), max_model_len=32_000)

        if normalized_model_id in _vllm_a100_80gb_128k_model_lower:
            return VllmContainerA100_80G(str(model_path), max_model_len=128_000)

        if normalized_model_id in _vllm_mid_models_lower:
            return VllmContainerA100_40G(str(model_path))

        if normalized_model_id in _vllm_top_model_lower:
            return VllmContainerA100_80G(str(model_path))
        
        # if normalized_model_id in _vllm_awq_models_lower:
        #     return VllmAWQ(str(model_path))

    raise ValueError(f"Invalid model: {model}")
