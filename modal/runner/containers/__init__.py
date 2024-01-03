from shared.protocol import ContainerType

from .vllm_unified import (
    VllmContainer_7B,
    VllmContainerA100_40G,
    VllmContainerA100_80G,
    VllmContainerA100_160G,
)

DEFAULT_CONTAINER_TYPES = {
    "mistralai/Mistral-7B-Instruct-v0.1": ContainerType.VllmContainer_7B,
    "HuggingFaceH4/zephyr-7b-beta": ContainerType.VllmContainer_7B,
    "Intel/neural-chat-7b-v3-1": ContainerType.VllmContainer_7B,
    "Undi95/Toppy-M-7B": ContainerType.VllmContainer_7B,
    "PygmalionAI/mythalion-13b": ContainerType.VllmContainerA100_40G,
    "Gryphe/MythoMax-L2-13b": ContainerType.VllmContainerA100_40G,
    "Undi95/ReMM-SLERP-L2-13B": ContainerType.VllmContainerA100_40G,
    "meta-llama/Llama-2-13b-chat-hf": ContainerType.VllmContainerA100_40G,
    "NousResearch/Nous-Hermes-Llama2-13b": ContainerType.VllmContainerA100_40G,
    "microsoft/Orca-2-13b": ContainerType.VllmContainerA100_40G,
    "jebcarter/Psyfighter-13B": ContainerType.VllmContainerA100_40G,
    "KoboldAI/LLaMA2-13B-Psyfighter2": ContainerType.VllmContainerA100_40G,
    "NeverSleep/Noromaid-20b-v0.1.1": ContainerType.VllmContainerA100_80G,
    "NousResearch/Nous-Capybara-34B": ContainerType.VllmContainerA100_80G_32K,
    "NousResearch/Yarn-Mistral-7b-128k": ContainerType.VllmContainerA100_80G_128K,
    # "eastwind/tinymix-8x1b-chat": ContainerType.VllmContainerA100_160G,
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b": ContainerType.VllmContainerA100_160G_8K,
    "cognitivecomputations/dolphin-2.7-mixtral-8x7b": ContainerType.VllmContainerA100_160G_8K,
    "NeverSleep/Noromaid-v0.1-mixtral-8x7b-Instruct-v3": ContainerType.VllmContainerA100_160G_8K,
}


def get_container(name: str, container_type: ContainerType):
    match container_type:
        case ContainerType.VllmContainer_7B:
            return VllmContainer_7B(name)
        case ContainerType.VllmContainerA100_40G:
            return VllmContainerA100_40G(name)
        case ContainerType.VllmContainerA100_80G:
            return VllmContainerA100_80G(name)
        case ContainerType.VllmContainerA100_80G_32K:
            return VllmContainerA100_80G(name, max_model_len=32_000)
        case ContainerType.VllmContainerA100_80G_128K:
            return VllmContainerA100_80G(name, max_model_len=128_000)
        case ContainerType.VllmContainerA100_160G:
            return VllmContainerA100_160G(name)
        case ContainerType.VllmContainerA100_160G_8K:
            return VllmContainerA100_160G(name, max_model_len=8_192)
        case ContainerType.VllmContainerA100_160G_32K:
            return VllmContainerA100_160G(name, max_model_len=32_000)
