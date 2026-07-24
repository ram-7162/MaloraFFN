
import torch
import os
import json
import random
import bitsandbytes as bnb
from src.Layers import updating_layers
from src.dataset import get_dataloaders
from src.model import build_model_and_tokenizer
from src.MaloraLayer import MALoRADownProjLayer,DenseLoRADownProjLayer,SymmetricMoEDownProjLayer

from src.Router import TopKGatingRouter
from src.dataset import MALoRADataset  
from torch.utils.data import DataLoader 
import torch.optim as optim
import gc
from transformers import AutoConfig

JSONL_PATH = 'data/final_data.jsonl'

r1   = 32
r2  = 96
alpha   = 32.0
n_experts  = 3
BATCH_SIZE  = 1
MAX_LENGTHS = {
    "default": 512,
    "expert_0": 1024 
}
SMOKE_TEST  = False
DEBUG_GRAD_CHECK = False
EPOCHS      = 3
MODE        = "lora"    ## symmetric_moe   ## lora   ## malora
SEED        = 42 
LEARNING_RATE = 1.32e-4
WEIGHT_DECAY = 0.065
SAMPLES_PER_EXPERT = 4 if SMOKE_TEST else None


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def log_result(record, path="results/model_train.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_num_layers(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return config.num_hidden_layers


num_layers = get_num_layers("meta-llama/Meta-Llama-3-8B-Instruct") 

alternate_layers = list(range(0, num_layers, 2))   


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

    total_loss = task_loss +  aux_loss

    
    if torch.isnan(task_loss):
        raise RuntimeError("task_loss became NaN")
    if torch.isnan(aux_loss):
        raise RuntimeError("aux_loss became NaN")
    if torch.isnan(total_loss):
        raise RuntimeError("total_loss became NaN")

    
    total_loss.backward()

    if DEBUG_GRAD_CHECK:
        for layer in model.model.layers:
            if isinstance(layer.mlp, (MALoRADownProjLayer, SymmetricMoEDownProjLayer)):
                grad = layer.mlp.router.Wg.weight.grad
                if grad is None:
                    print("WARNING: router grad is None — not receiving signal")
                else:
                    print(f"router grad norm: {grad.norm().item():.6f}")

    
    torch.nn.utils.clip_grad_norm_(get_trainable_params(model), max_norm=1.0)
    optimizer.step()
    
    return total_loss.item(),task_loss.item(), aux_loss.item()


def val_step(model, batch, device):
    model.eval()
    with torch.no_grad():
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        task_loss = outputs.loss

        aux_loss = torch.tensor(0.0, device=device)
        for layer in model.model.layers:
            if isinstance(layer.mlp, (MALoRADownProjLayer, SymmetricMoEDownProjLayer)):
                if layer.mlp.last_auxloss is not None:
                    aux_loss = aux_loss + layer.mlp.last_auxloss

        return task_loss.item(), aux_loss.item()


def run_validation(model, val_dataloader, device):
    total_task, total_aux, n = 0.0, 0.0, 0
    for batch in val_dataloader:
        task_loss, aux_loss = val_step(model, batch, device)
        total_task += task_loss
        total_aux  += aux_loss
        n += 1
    return total_task / n, total_aux / n


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

    
    train_loader, val_loader = get_dataloaders(JSONL_PATH, tokenizer, batch_size=BATCH_SIZE, max_lengths=MAX_LENGTHS,seed=SEED, samples_per_expert=SAMPLES_PER_EXPERT)


    optimizer = bnb.optim.AdamW8bit(get_trainable_params(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*50}")
    print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"{'='*50}\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(train_loader):
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

        
        val_task_loss, val_aux_loss = run_validation(model, val_loader, device)
        print(f"  Epoch {epoch+1} | val_task_loss={val_task_loss:.4f} | val_aux_loss={val_aux_loss:.4f}")

        log_result({
            "stage": MODE,
            "epoch": epoch + 1,
            "val_task_loss": val_task_loss, "val_aux_loss": val_aux_loss,
        })


        if not SMOKE_TEST:
            torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pt')
            print(f"  Checkpoint saved → checkpoints/epoch_{epoch+1}.pt")

        
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()




run()
