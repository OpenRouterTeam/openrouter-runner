from .vllm_unified import (
    IntelNeuralChat7B,
    JebCarterPsyfighter13B,
    JohnDurbinBagel34B,
    KoboldAIPsyfighter2,
    MicrosoftPhi2,
    NeverSleepNoromaidMixtral8x7B,
)

DEFAULT_CONTAINERS = {
    "microsoft/phi-2": MicrosoftPhi2,
    "Intel/neural-chat-7b-v3-1": IntelNeuralChat7B,
    "jebcarter/Psyfighter-13B": JebCarterPsyfighter13B,
    "KoboldAI/LLaMA2-13B-Psyfighter2": KoboldAIPsyfighter2,
    "NeverSleep/Noromaid-v0.1-mixtral-8x7b-Instruct-v3": NeverSleepNoromaidMixtral8x7B,
    "jondurbin/bagel-34b-v0.2": JohnDurbinBagel34B,
}
