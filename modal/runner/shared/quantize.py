import modal.gpu
from modal import Secret

from runner.shared.common import stub
from shared.images import BASE_IMAGE
from shared.logging import get_logger, get_observability_secrets
from shared.volumes import (
    does_model_exist,
    get_model_path,
    models_path,
    models_volume,
)

logger = get_logger(__name__)


quantizer_image = (
    BASE_IMAGE.apt_install("git")
    .pip_install("auto-gptq==0.7.1")
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("huggingface_hub==0.20.2")
    .pip_install("hf-transfer==0.1.4")
    .pip_install("transformers==4.39.1")
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
def quantize_model(
    pretrained: str = "sophosympatheia/Midnight-Rose-70B-v2.0.3",
    quantized: str = "sambarnes/Midnight-Rose-70B-v2.0.3-GPTQ",
):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    pretrained_model_path = get_model_path(pretrained)
    quantized_model_path = get_model_path(quantized)
    if not does_model_exist(pretrained_model_path):
        logger.error("Pretrained does not exist at", pretrained_model_path)
        return
    elif does_model_exist(quantized_model_path):
        logger.info("Quantized model already exists at", quantized_model_path)
        return

    logger.info("Loading calibration dataset...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        use_fast=True,
    )
    examples = load_open_instruct(tokenizer=tokenizer)
    examples_for_quant = [
        {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
        }
        for example in examples
    ]

    # By default, the model will always be loaded into CPU memory
    logger.info("Loading pretrained model into memory...")
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_path,
        quantize_config=BaseQuantizeConfig(
            bits=4,
            desc_act=False,  # False can significantly speed up inference but might slightly decrease perplexity
            damp_percent=0.1,  # Affects how samples are processed, 0.1 results in slightly better accuracy
        ),
    )

    # Examples are dict whose keys can only be "input_ids" and "attention_mask".
    # This is called the "GPTQ dataset" in TheBloke's READMEs, where he says:
    #
    #   "The calibration dataset used during quantisation. Using a dataset more
    #    appropriate to the model's training can improve quantisation accuracy.
    #    Note that the GPTQ calibration dataset is not the same as the dataset
    #    used to train the model - please refer to the original model repo for
    #    details of the training dataset(s)."

    logger.info("Starting quantization...")
    model.quantize(examples_for_quant)
    model.save_quantized(quantized_model_path, use_safetensors=True)

    # TODO: copy the tokenizer files as well? I just did it manually for now

    logger.info("Committing to volume...")
    models_volume.commit()
    logger.info(f"Volume now contains {quantized_model_path}")


def load_open_instruct(tokenizer, n_samples=128):
    import torch
    from datasets import Dataset, load_dataset

    # TODO: maybe randomize the sampling
    train_split = load_dataset("VMWare/open-instruct")["train"]
    model_max_length = tokenizer.model_max_length

    def dummy_gen():
        for i, x in enumerate(train_split):
            if i == n_samples:
                break
            yield x

    def tokenize(examples):
        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for prompt, response in zip(
            examples["alpaca_prompt"], examples["response"]
        ):
            text = prompt + response
            if len(tokenizer(prompt)["input_ids"]) >= model_max_length:
                continue

            data = tokenizer(text)
            input_ids.append(data["input_ids"][:model_max_length])
            attention_mask.append(data["attention_mask"][:model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset
