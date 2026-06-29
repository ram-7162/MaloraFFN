import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.Layers import updating_layers

load_dotenv() 

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")
HF_TOKEN   = os.getenv("HF_TOKEN")  # your Hugging Face access token


r1          = 64
r2          = 128
alpha       = 16.0
n_experts   = 3
BATCH_SIZE  = 4
MAX_LENGTH  = 512

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(r1, r2, alpha, n_experts, layer_range=(8, 24), dtype=torch.float32):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        token=HF_TOKEN,
    )
    # (model, r1, r2, alpha, n_experts
    model = updating_layers(model, r1, r2, alpha, n_experts)
    return model


def build_model_and_tokenizer(r1, r2, alpha, n_experts, layer_range=(8, 24), dtype=torch.float32):
    tokenizer = load_tokenizer()
    model = load_model(r1, r2, alpha, n_experts, layer_range=layer_range, dtype=dtype)
    return model, tokenizer