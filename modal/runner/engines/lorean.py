from modal import method
from pydantic import BaseModel

from shared.protocol import (
    Payload,
    create_error_text,
    create_sse_data,
)
from shared.volumes import get_lora_path, get_model_path

from .base import BaseEngine


# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L192
class LoreanParams(BaseModel):
    model: str


class LoreanEngine(BaseEngine):
    def __init__(self, params: LoreanParams):
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

        base_model = "mistralai/Mistral-7B-Instruct-v0.1"
        model_path = get_model_path(base_model).resolve().absolute()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            local_files_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
            local_files_only=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_model_len = 512

    # @method()
    # async def tokenize_prompt(self, payload: Payload) -> List[int]:
    #     return self.tokenizer.encode(payload.prompt)

    # @method()
    # async def max_model_len(self) -> int:
    #     return self.max_model_len

    @method()
    async def generate(self, payload: Payload, params, input_ids):
        try:
            # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/generation/streamers.py#L182-L203
            import time
            from threading import Thread

            from peft import PeftModel
            from transformers import TextIteratorStreamer

            finetune_id = "viggo-finetune"
            user_name = "lab"
            model_inputs = self.tokenizer(
                payload.prompt, return_tensors="pt"
            ).to("cuda")

            streamer = TextIteratorStreamer(self.tokenizer)

            lora_path = get_lora_path(user_name, finetune_id)

            ft_model = PeftModel.from_pretrained(
                self.model, lora_path.resolve().absolute()
            )
            ft_model.generate(
                **model_inputs, max_new_tokens=256, pad_token_id=2
            )
            generation_kwargs = dict(
                inputs=model_inputs, streamer=streamer, max_new_tokens=256
            )

            thread = Thread(target=ft_model.generate, kwargs=generation_kwargs)
            thread.start()
            t0 = time.time()
            tokens_count = 0
            output = ""
            for token in streamer:
                # Skipping invalid UTF8 tokens:
                if token and token == "\ufffd":
                    continue
                if payload.stream:
                    yield create_sse_data(token)
                else:
                    output += token
                # Token accounting
                tokens_count += 1

            if not payload.stream:
                yield create_sse_data(output)

            throughput = tokens_count / (time.time() - t0)
            print(f"Tokens count: {tokens_count} tokens")
            print(f"Request completed: {throughput:.4f} tokens/s")

            # yield "[DONE]"
            # print(request_output.outputs[0].text)
        except Exception as err:
            e = create_error_text(err)
            print(e)
            if payload.stream:
                yield create_sse_data(e)
            else:
                yield e
