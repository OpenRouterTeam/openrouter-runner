import modal.gpu
from modal import Secret

from runner.shared.common import stub
from shared.images import BASE_IMAGE
from shared.logging import get_logger, get_observability_secrets
from shared.volumes import (
    get_model_path,
    models_path,
    models_volume,
)

logger = get_logger(__name__)

cache_path = get_model_path("__cache__")


quantizer_image = (
    BASE_IMAGE.apt_install("git")
    .pip_install("auto-gptq==0.7.1")
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("huggingface_hub==0.20.2")
    .pip_install("hf-transfer==0.1.4")
    .pip_install("transformers==4.31.0")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
)


@stub.function(
    image=quantizer_image,
    gpu=modal.gpu.H100(count=1),
    volumes={models_path: models_volume},
    secrets=[Secret.from_name("huggingface"), *get_observability_secrets()],
    timeout=3600 * 3,  # 3 hours
)
def quantize_model():
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    # TODO: make these configurable on the request
    pretrained_model_path = get_model_path(
        "sophosympatheia/Midnight-Rose-70B-v2.0.3"
    )
    quantized_model_path = get_model_path(
        "sambarnes/Midnight-Rose-70B-v2.0.3-GPTQ-naive"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path, use_fast=True
    )
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_path, quantize_config
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_path, use_safetensors=True)
    models_volume.commit()
    logger.info(f"Volume now contains {quantized_model_path}")
