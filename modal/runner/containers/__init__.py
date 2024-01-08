from shared.protocol import ContainerType

from .vllm_unified import (
    VllmContainer_7B,
    VllmContainerA100_40G,
    VllmContainerA100_80G,
    VllmContainerA100_160G,
    VllmContainerA100_160G_Isolated,
)

DEFAULT_CONTAINER_TYPES = {
    "Intel/neural-chat-7b-v3-1": ContainerType.VllmContainer_7B,
    "PygmalionAI/mythalion-13b": ContainerType.VllmContainerA100_40G,
    "jebcarter/Psyfighter-13B": ContainerType.VllmContainerA100_40G,
    "KoboldAI/LLaMA2-13B-Psyfighter2": ContainerType.VllmContainerA100_40G,
    "Austism/chronos-hermes-13b-v2": ContainerType.VllmContainerA100_40G,
    "NeverSleep/Noromaid-20b-v0.1.1": ContainerType.VllmContainerA100_80G,
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b": ContainerType.VllmContainerA100_160G,
    "NeverSleep/Noromaid-v0.1-mixtral-8x7b-Instruct-v3": ContainerType.VllmContainerA100_160G,
    "jondurbin/bagel-34b-v0.2": ContainerType.VllmContainerA100_160G_Isolated,
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
        case ContainerType.VllmContainerA100_160G:
            return VllmContainerA100_160G(name)
        case ContainerType.VllmContainerA100_160G:
            return VllmContainerA100_160G(name)
        case ContainerType.VllmContainerA100_160G_Isolated:
            return VllmContainerA100_160G_Isolated(name)
