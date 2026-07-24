import torch
from transformers import AutoConfig
from src.model import build_model_and_tokenizer
from lm_eval.api.model import LM
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
CHECKPOINT_PATH = "checkpoints/epoch_3.pt"   
MODE = "lora"
R1 = 32
R2 = 96
ALPHA = 32.0
N_EXPERTS = 3
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_num_layers(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return config.num_hidden_layers


def load_trained_model():
    num_layers = get_num_layers(MODEL_NAME)
    alternate_layers = list(range(0, num_layers, 2))

    model, tokenizer = build_model_and_tokenizer(
        r1=R1, r2=R2, alpha=ALPHA, n_experts=N_EXPERTS,
        layer_range=alternate_layers, mode=MODE
    )

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[SUCCESS] Loaded adapter weights! Missing base keys: {len(missing)} (Expected)")
    
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH} (mode={MODE})")
    return model, tokenizer

class MaloraLMWrapper(LM):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model=model
        self.tokenizer=tokenizer
        self._device=device
    def generate_until(self,requests):
        results=[]
        for req in requests:
            prompt=req.args[0]
            inputs=self.tokenizer(prompt, return_tensors="pt").to(self._device)
            output=self.model.generate(**inputs, max_new_tokens=256)
            generate_ids=output[0][inputs["input_ids"].shape[1]:]

            text=self.tokenizer.decode(generate_ids, skip_special_tokens=True)
            gen_kwargs=req.args[1]

            until=gen_kwargs.get("until",[])
            if until:
                for u in until:
                    if u in text:
                        text=text.split(u)[0]
            results.append(text)
        return results
    def loglikelihood(self, requests):
        raise NotImplementedError("Not needed for generation based tasks")
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Not needed for generation based tasks")


import lm_eval

import json, os

if __name__ == "__main__":
    model, tokenizer = load_trained_model()
    wrapper = MaloraLMWrapper(model, tokenizer, device="cuda")

    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=["humaneval"],
        limit=5,
        confirm_run_unsafe_code=True,
    )

    print(results["results"])

    os.makedirs("results", exist_ok=True)
    with open(f"results/{MODE}_eval.json", "w") as f:
        json.dump(results["results"], f, indent=2)