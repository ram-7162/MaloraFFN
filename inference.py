import torch
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging

logging.set_verbosity_error()
from src.model import build_model_and_tokenizer


# Same hyperparameters used during training
r1 = 64
r2 = 128
alpha = 16.0
n_experts = 3

CHECKPOINT = "checkpoints/epoch1_step19000.pt"

# Build model exactly like training
model, tokenizer = build_model_and_tokenizer(
    r1,
    r2,
    alpha,
    n_experts,
    layer_range=(8, 24)
)

# Load checkpoint
state_dict = torch.load(CHECKPOINT, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.eval()

device = next(model.parameters()).device


def generate(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return answer


while True:
    print("\nEnter your prompt (type END on a new line when finished):")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    prompt = "\n".join(lines)

    if prompt.strip().lower() == "exit":
        break

    print("\nResponse:\n")
    print(generate(prompt))