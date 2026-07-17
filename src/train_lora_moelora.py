
import torch
import os
import bitsandbytes as bnb
from src.dataset import get_dataloader
from src.Layers import updating_layers
from src.model import build_model_and_tokenizer
from src.MaloraLayer import MALoRADownProjLayer,SymmetricMoEDownProjLayer
import torch.optim as optim

JSONL_PATHS = {
    0: 'data/expert0_algo_training.jsonl',
    1: 'data/expert1_syntax_training.jsonl',
    2: 'data/expert2_secure_training.jsonl',
}

r1   = 32
r2  = 96
alpha   = 32.0
n_experts  = 3
BATCH_SIZE  = 1
MAX_LENGTHS = {
    "default": 512,
    "expert_0": 1024 
}
SMOKE_TEST  = True
EPOCHS      = 3
MODE        = "malora"
SEED        = 42 
LEARNING_RATE = 1.32e-4
WEIGHT_DECAY = 0.065
SAMPLES_PER_EXPERT = 4 if SMOKE_TEST else None


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def train_step(model, batch, optimizer, device):
    model.train()
    optimizer.zero_grad()

    input_ids      = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels         = batch['labels'].to(device)      

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    task_loss = outputs.loss
    aux_loss = torch.tensor(0.0, device=device)
    
    for layer in model.model.layers:
        if isinstance(layer.mlp, (MALoRADownProjLayer, SymmetricMoEDownProjLayer)):
            if layer.mlp.last_auxloss is not None:
                aux_loss = aux_loss + layer.mlp.last_auxloss

    total_loss = task_loss + 0.01 * aux_loss

    
    if torch.isnan(task_loss):
        raise RuntimeError("task_loss became NaN")
    if torch.isnan(aux_loss):
        raise RuntimeError("aux_loss became NaN")
    if torch.isnan(total_loss):
        raise RuntimeError("total_loss became NaN")

    
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(get_trainable_params(model), max_norm=1.0)
    optimizer.step()
    
    return total_loss.item(),task_loss.item(), aux_loss.item()


def run():
    
    model, tokenizer = build_model_and_tokenizer(
            r1=32, r2=96, alpha=alpha, n_experts=n_experts,
            layer_range=alternate_layers, mode=MODE
        )

    device = torch.device("cuda")

    for module in model.modules():
        if isinstance(module, TopKGatingRouter):
            module.to(device).float() # Ensure router is float32 and on GPU
        elif isinstance(module, (MALoRADownProjLayer, SymmetricMoEDownProjLayer)):
            module.to(device)
            
    dataloader = get_dataloader(
        JSONL_PATHS,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        samples_per_expert=SAMPLES_PER_EXPERT
    )

    optimizer = bnb.optim.AdamW8bit(
    get_trainable_params(model),
    lr=1e-4,
        )

    print(f"\n{'='*50}")
    print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
    print(f"Total batches per epoch: {len(dataloader)}")
    print(f"{'='*50}\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(dataloader):
            total_loss,task_loss, aux_loss = train_step(model, batch, optimizer, device)
            if (step+1)%100==0 or step<20:
                print(
                    f"Step {step+1} | "
                    f"total_loss={total_loss:.4f} | "
                    f"task_loss={task_loss:.4f} | "
                    f"aux_loss={aux_loss:.4f}"
                    )
            if (step + 1) % 1000 == 0:
                os.makedirs("checkpoints", exist_ok=True)

                torch.save(
                model.state_dict(),
                f"checkpoints/epoch{epoch+1}_step{step+1}.pt"
                )

                print(f"Checkpoint saved at epoch {epoch+1}, step {step+1}")

        if not SMOKE_TEST:
            torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pt')
            print(f"  Checkpoint saved → checkpoints/epoch_{epoch+1}.pt")

    print("\nDone!")
