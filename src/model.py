import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from src.Layers import updating_layers
from src.Layers import updating_layers_alternative
from peft import prepare_model_for_kbit_training


load_dotenv() 

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN   = os.getenv("HF_TOKEN")  
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


r1          = 32
r2          = 96
alpha       = 32.0
n_experts   = 3
BATCH_SIZE  = 1
MAX_LENGTHS = {
    "default": 512,
    "expert_0": 1024 
}

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(r1, r2, alpha, n_experts, layer_range, dtype=torch.float16, mode="malora"):
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    )
        # (model, r1, r2, alpha, n_experts
   
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if isinstance(layer_range, tuple):
        model = updating_layers(model, r1, r2, alpha, n_experts, layer_range, mode)
    elif isinstance(layer_range, list):
        model = updating_layers_alternative(model, r1, r2, alpha, n_experts, layer_range, mode)
    else:
        raise TypeError("layer_range must be either tuple or list")

    return model

def build_model_and_tokenizer(r1, r2, alpha, n_experts, layer_range=(8, 24), dtype=torch.float16, mode="malora"):
    tokenizer = load_tokenizer()
    model = load_model(r1, r2, alpha, n_experts, layer_range=layer_range, dtype=dtype, mode=mode)
    return model, tokenizer
