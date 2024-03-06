from .vllm_unified import (
    VllmContainer_IntelNeuralChat7B,
    VllmContainer_JebCarterPsyfighter13B,
    VllmContainer_JohnDurbinBagel34B,
    VllmContainer_KoboldAIPsyfighter2,
    VllmContainer_MicrosoftPhi2,
    VllmContainer_NeverSleepNoromaidMixtral8x7B,
)

DEFAULT_CONTAINERS = {
    "microsoft/phi-2": VllmContainer_MicrosoftPhi2,
    "Intel/neural-chat-7b-v3-1": VllmContainer_IntelNeuralChat7B,
    "jebcarter/Psyfighter-13B": VllmContainer_JebCarterPsyfighter13B,
    "KoboldAI/LLaMA2-13B-Psyfighter2": VllmContainer_KoboldAIPsyfighter2,
    "NeverSleep/Noromaid-v0.1-mixtral-8x7b-Instruct-v3": VllmContainer_NeverSleepNoromaidMixtral8x7B,
    "jondurbin/bagel-34b-v0.2": VllmContainer_JohnDurbinBagel34B,
}
