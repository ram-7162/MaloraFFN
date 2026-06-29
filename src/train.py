# import torch.optim as optim
# from src.Layers import updating_layers
# from src.Layers import model
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from src.MaloraLayer import MALoRADownProjLayer


# def get_trainable_params(model):
#     return [p for p in model.parameters() if p.requires_grad]


# optimizer = optim.AdamW(get_trainable_params(model), lr=3e-4)
# # criterion = nn.CrossEntropyLoss()

# def train_step(model, batch):
#     model.train()
#     optimizer.zero_grad()

#     # output = model.forward(batch_features)
#     # task_loss = criterion(output, batch_labels)  

#     outputs = model(
#         input_ids=batch["input_ids"],
#         attention_mask=batch["attention_mask"],
#         labels=batch["input_ids"]   # next token prediction
#     )
#     task_loss = outputs.loss


#     # aux_loss = torch.tensor(0.0, device=model.device)
#     aux_loss = torch.tensor(0.0, device=next(model.parameters()).device)
#     for layer in model.model.layers:
#         if isinstance(layer.mlp, MALoRADownProjLayer):
#             aux_loss = aux_loss + layer.mlp.last_auxloss 

#     total_loss = task_loss + aux_loss

#     total_loss.backward()
#     optimizer.step()

#     return task_loss.item(), aux_loss.item()


# def train(model, dataloader, epochs=3):
    

#     for epoch in range(epochs):
#         print(f"epoch : {epoch}")
#         for batch in dataloader:
#             task_loss, aux_loss = train_step(model,batch)
#             print(f"task={task_loss:.4f}  aux={aux_loss:.4f}")




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset import get_dataloader
from src.Layers import updating_layers
from src.MaloraLayer import MALoRADownProjLayer
import torch.optim as optim

# ── Config ───────────────────────────────────────────────
JSONL_PATHS = {
    0: 'data/expert0_algo_training.jsonl',
    1: 'data/expert1_syntax_training.jsonl',
    2: 'data/expert2_secure_training.jsonl',
}

r1          = 64
r2          = 128
alpha       = 16.0
n_experts   = 3
BATCH_SIZE  = 4
MAX_LENGTH  = 512
EPOCHS      = 3
SMOKE_TEST  = True   # ← set False for full training

SAMPLES_PER_EXPERT = 4 if SMOKE_TEST else None   # 4 samples × 3 experts = 12 total


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token   # Llama has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float32
)

model = updating_layers(model, r1, r2, alpha, n_experts, layer_range=(8, 24))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)

# ── Dataloader ───────────────────────────────────────────
dataloader = get_dataloader(
    JSONL_PATHS,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    samples_per_expert=SAMPLES_PER_EXPERT
)

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

optimizer = optim.AdamW(get_trainable_params(model), lr=3e-4)

def train_step(model, batch):
    model.train()
    optimizer.zero_grad()

    input_ids      = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )
    task_loss = outputs.loss

    aux_loss = torch.tensor(0.0, device=device)
    for layer in model.model.layers:
        if isinstance(layer.mlp, MALoRADownProjLayer):
            if layer.mlp.last_auxloss is not None:
                aux_loss = aux_loss + layer.mlp.last_auxloss

    total_loss = task_loss + 0.01 * aux_loss   # aux weight = 0.01

    total_loss.backward()
    optimizer.step()

    return task_loss.item(), aux_loss.item()

print(f"\n{'='*50}")
print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
print(f"Total batches per epoch: {len(dataloader)}")
print(f"{'='*50}\n")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(dataloader):
        task_loss, aux_loss = train_step(model, batch)
        print(f"  Step {step+1} | task_loss={task_loss:.4f} | aux_loss={aux_loss:.4f}")

    # Save checkpoint after each epoch
    if not SMOKE_TEST:
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pt')
        print(f"  Checkpoint saved → checkpoints/epoch_{epoch+1}.pt")

print("\nDone!")