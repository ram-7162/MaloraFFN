import sys
import os
import torch
import itertools

# Allow imports from the cloned MultiPL-E repository
sys.path.insert(0, os.path.expanduser("~/MultiPL-E"))

from multipl_e.completions import (
    make_main,
    stop_at_stop_token,
    partial_arg_parser,
)

from src.model import build_model_and_tokenizer
from transformers import AutoConfig


CHECKPOINT_PATH = "checkpoints/epoch_3.pt"

MODE = "lora"

R1 = 32
R2 = 96
ALPHA = 32.0
N_EXPERTS = 3

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_trained_model():
    print(f"Loading base model: {MODEL_NAME}")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers

    # MUST match the layer configuration used during LoRA training
    alternate_layers = list(range(0, num_layers, 2))

    model, tokenizer = build_model_and_tokenizer(
        r1=R1,
        r2=R2,
        alpha=ALPHA,
        n_experts=N_EXPERTS,
        layer_range=alternate_layers,
        mode=MODE,
    )

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if unexpected:
        print("Unexpected key examples:")
        for key in unexpected[:10]:
            print(" ", key)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("[SUCCESS] LoRA checkpoint loaded")

    return model, tokenizer


class MaloraMultiPLEModel:

    def __init__(self):

        self.model, self.tokenizer = load_trained_model()

        self.device = next(
            self.model.parameters()
        ).device

        self.all_special_token_ids = (
            self.tokenizer.all_special_ids
        )

    def completion_tensors(
        self,
        prompts,
        max_length,
        temperature,
        top_p,
    ):

        self.model.eval()

        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_length - 1,
        ).to(self.device)

        generation_kwargs = {
            "max_length": max_length,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # MultiPL-E normally uses sampling.
        # Temperature 0 gives deterministic greedy generation.
        if temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        return outputs


    def remove_padding_and_special_tokens(
        self,
        token_ids,
    ):

        pad_id = self.tokenizer.pad_token_id
        bos_id = self.tokenizer.bos_token_id

        ids = list(token_ids)

        # Remove left padding / BOS
        while ids and (
            ids[0] == pad_id
            or (
                bos_id is not None
                and ids[0] == bos_id
            )
        ):
            ids.pop(0)

        # Stop at first special token
        cleaned = []

        for token_id in ids:

            if token_id in self.all_special_token_ids:
                break

            cleaned.append(token_id)

        return cleaned


    def decode_single_output(
        self,
        output_tensor,
        prompt,
    ):

        token_ids = self.remove_padding_and_special_tokens(
            output_tensor.tolist()
        )

        text = self.tokenizer.decode(
            token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

        # Remove original prompt
        if text.startswith(prompt):
            text = text[len(prompt):]

        return text


    def completions(
        self,
        prompts,
        max_tokens,
        temperature,
        top_p,
        stop,
    ):

        prompts = [
            prompt.strip()
            for prompt in prompts
        ]

        # Calculate required total sequence length
        encoded = self.tokenizer(
            prompts,
            padding=False,
            return_tensors=None,
        )

        longest_prompt = max(
            len(ids)
            for ids in encoded["input_ids"]
        )

        max_length = longest_prompt + max_tokens

        output_tensors = self.completion_tensors(
            prompts,
            max_length,
            temperature,
            top_p,
        )

        results = []

        for prompt, output_tensor in zip(
            prompts,
            output_tensors,
        ):

            text = self.decode_single_output(
                output_tensor,
                prompt,
            )

            text = stop_at_stop_token(
                text,
                stop,
            )

            results.append(text)

        return results


def main():

    parser = partial_arg_parser()

    # automodel.py normally requires --name.
    # We only use it as an output label.
    parser.add_argument(
        "--name",
        type=str,
        default="lora",
    )

    args = parser.parse_args()

    model = MaloraMultiPLEModel()

    make_main(
        args,
        "lora",
        model.completions,
    )


if __name__ == "__main__":
    main()