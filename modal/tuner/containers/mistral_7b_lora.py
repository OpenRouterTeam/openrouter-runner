# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import Image, Secret, gpu, method

from shared.protocol import create_error_text, create_sse_data
from shared.volumes import (
    get_lora_path,
    get_model_path,
    loras_path,
    models_path,
)
from tuner.shared.common import stub

# TODO: Swap to lower-end GPU on prod
_gpu = gpu.A100(count=1, memory=80)

_vllm_image = Image.from_registry(
    "nvcr.io/nvidia/pytorch:23.09-py3"
    # "nvcr.io/nvidia/pytorch:22.12-py3"
).pip_install(
    "bitsandbytes",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "scipy",
    "ipywidgets",
    "wandb",
)


# DEV function to test out the setup, should be replaced with HF's tokenizer chat_template
def format_data_point(data_point):
    return f"""
Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

### Target sentence:
{data_point["target"]}

### Meaning representation:
{data_point["meaning_representation"]}
"""


@stub.cls(
    secrets=[
        Secret.from_name("wandb"),
        Secret.from_name("huggingface"),
    ],
    volumes={
        str(loras_path): stub.loras_volume,
        str(models_path): stub.models_volume,
    },
    image=_vllm_image,
    gpu=_gpu,
    timeout=60 * 60 * 5,  # 5 hours
    container_idle_timeout=1200,  # 20 minutes
    # cpu=8,
    # memory=32,
)
class Mistral7BLoraContainer:
    def __init__(self):
        pass

    async def __aenter__(self):
        # TODO: Move the base_model_id to init params
        base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.base_model_id = base_model_id

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_path = get_model_path(base_model_id).resolve().absolute()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            local_files_only=True,
        )
        self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
            local_files_only=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @method()
    async def generate(self):
        try:
            import wandb

            yield create_sse_data("Init logger...")

            wandb.init(
                project="tuner",
            )

            from accelerate import Accelerator, FullyShardedDataParallelPlugin
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullOptimStateDictConfig,
                FullStateDictConfig,
            )

            yield create_sse_data("Loading accelerator...")

            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True,
                    rank0_only=False,
                ),
                optim_state_dict_config=FullOptimStateDictConfig(
                    offload_to_cpu=True,
                    rank0_only=False,
                ),
            )

            accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

            from datasets import load_dataset

            # TODO: Upload data set to Supabase, then download?
            # train_dataset = load_dataset(
            #     "json", data_files="notes.jsonl", split="train"
            # )
            # eval_dataset = load_dataset(
            #     "json", data_files="notes_validation.jsonl", split="train"
            # )

            yield create_sse_data("Loading data set...")

            train_dataset = load_dataset("gem/viggo", split="train")
            eval_dataset = load_dataset("gem/viggo", split="validation")

            # TODO: use tokenizer.apply_chat_template to leverage HF's tokenizer chat_template:
            def tokenize(prompt):
                # TODO: use an ChatML[][] to leverage HF's tokenizer chat_template:
                # encodeds = tokenizer.apply_chat_template(
                #     payload.messages, return_tensors="pt"
                # )

                result = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                )
                result["labels"] = result["input_ids"].copy()
                return result

            # Map through each data in the set, and call format_data_point before passing it to tokenize
            tokenized_train_dataset = train_dataset.map(
                lambda data: tokenize(format_data_point(data))
            )

            tokenized_eval_dataset = eval_dataset.map(
                lambda data: tokenize(format_data_point(data))
            )

            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            yield create_sse_data("Prepare model...")

            model = prepare_model_for_kbit_training(self.model)

            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                bias="none",
                lora_dropout=0.05,  # Conventional
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            # Apply the accelerator. You can comment this out to remove the accelerator.
            model = accelerator.prepare(model)

            yield create_sse_data("Accelerator applied...")

            import torch

            if torch.cuda.device_count() > 1:  # If more than 1 GPU
                yield create_sse_data("parallel")
                model.is_parallelizable = True
                model.model_parallel = True

            from datetime import datetime

            from transformers import (
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )

            # TODO: Move to generator params
            finetune_id = "viggo-finetune"
            user_name = "lab"

            lora_path = get_lora_path(user_name, finetune_id)

            trainer = Trainer(
                model=model,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_eval_dataset,
                args=TrainingArguments(
                    output_dir=str(lora_path),
                    warmup_steps=5,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    max_steps=1000,
                    learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
                    logging_steps=50,
                    bf16=True,
                    optim="paged_adamw_8bit",
                    save_strategy="steps",  # Save the model checkpoint every logging step
                    save_steps=100,  # Save checkpoints every 50 steps
                    evaluation_strategy="steps",  # Evaluate the model every logging step
                    eval_steps=100,  # Evaluate and save checkpoints every 50 steps
                    do_eval=True,  # Perform evaluation at the end of training
                    report_to="wandb",  # Comment this out if you don't want to use weights & baises
                    run_name=f"{user_name} | {finetune_id} | {self.base_model_id} | {datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
                ),
                data_collator=DataCollatorForLanguageModeling(
                    self.tokenizer,
                    mlm=False,
                ),
            )

            model.config.use_cache = False
            yield create_sse_data("Begin training...")

            trainer.train()
            yield create_sse_data("Training finished.")

        except Exception as err:
            e = create_error_text(err)
            print(e)
            yield create_sse_data(e)
