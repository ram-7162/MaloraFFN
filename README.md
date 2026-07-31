# MaloraFFN — MALORA Ablation Training

Training code for **MALORA (Mixture of Asymmetric Low-Rank Adapters)** — a Mixture-of-Experts LoRA architecture applied to the FFN (down-projection) layers of a base LLM, with two other ablation modes (`lora`, `symmetric_moe`) for comparison.

This repo lets you fine-tune a base causal LM with one of three FFN adapter architectures, using per-expert training data (AlgoLogic, SyntaxPoly, SecureDebug), and save periodic checkpoints for evaluation.

## 1. What's in here

```
main.py                    # entrypoint -> calls src/train.py:run()
src/
  model.py                 # loads base model in 4-bit (bnb) + swaps in adapter layers
  Layers.py                # updating_layers(): replaces model.model.layers[i].mlp with an adapter
  MaloraLayer.py            # MALoRADownProjLayer, DenseLoRADownProjLayer, SymmetricMoEDownProjLayer
  Router.py                 # TopKGatingRouter (used inside MALoRADownProjLayer)
  Experts.py, SharedSubspace.py
  dataset.py                 # MALoRADataset + get_dataloaders() (multi-file, per-expert jsonl)
  train.py                   # training loop used by main.py (single combined dataloader)
  train_lora_moelora.py      # alternate training script w/ train/val split + logging (this is the one we've been running)
  evaluate.py                 # eval harness helpers
inference.py                # simple REPL to chat with a trained checkpoint
Multipl.py                  # MultiPL-E benchmark generation script
requirements.txt
data/                        # expert training jsonl files
checkpoints/                 # saved .pt state dicts (gitignored)
results/                     # eval outputs + training logs (jsonl)
```


Make sure the `JSONL_PATHS` / data file referenced in whichever script you use actually exists before launching a full run.

## 2. Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (the model is loaded 4-bit via `bitsandbytes`; `train.py`/`train_lora_moelora.py` hardcode `torch.device("cuda")`).

Create a `.env` file in the repo root (loaded via `python-dotenv` in `src/model.py`):

```
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct   # or Qwen/Qwen2.5-Coder-7B-Instruct, etc.
HF_TOKEN=hf_xxx                                   # needed for gated models like Llama-3
```

If you don't set `MODEL_NAME`, it defaults to `meta-llama/Meta-Llama-3-8B-Instruct` in `src/model.py`. Note `src/evaluate.py` and `Multipl.py` currently hardcode `MODEL_NAME` at the top of the file rather than reading from `.env` — update those manually if you switch base models.

## 3. Data format

Each line of an expert jsonl file is:

```json
{"instruction": "...", "input": "", "output": "..."}
```

`instruction`/`input` are combined into the user turn, `output` becomes the assistant turn, formatted through the tokenizer's chat template (`dataset.py: format_prompt`). Loss is masked on everything up to the end of the prompt (`labels[:prompt_len] = -100`).

- `data/expert0_algo_training.jsonl`, `expert1_syntax_training.jsonl`, `expert2_secure_training.jsonl` — per-expert files consumed by `src/train.py` (AlgoLogic / SyntaxPoly / SecureDebug).
- `data/final_data.jsonl` — combined file (with an `expert_id` field per row) consumed by `src/train_lora_moelora.py`.

## 4. Choosing the adapter mode

Set `MODE` in whichever training script you're using:

| Mode | Class | What it does |
|---|---|---|
| `"malora"` | `MALoRADownProjLayer` | Asymmetric-rank Mixture-of-LoRA on the FFN down-proj, with a `TopKGatingRouter` selecting experts per token and an auxiliary load-balancing loss |
| `"symmetric_moe"` | `SymmetricMoEDownProjLayer` | Same MoE routing but all experts use the same rank |
| `"lora"` | `DenseLoRADownProjLayer` | A single dense LoRA adapter, no routing (baseline) |

`Layers.py:updating_layers()` swaps `model.model.layers[i].mlp` for the chosen adapter class on every layer index in `layer_range`. In `train_lora_moelora.py`, `layer_range` is computed as **every 2nd layer** (`alternate_layers = list(range(0, num_layers, 2))`) 

Key hyperparameters (set near the top of the training script):
- `r1`, `r2` — LoRA ranks (MALORA uses two asymmetric ranks; `lora`/`symmetric_moe` mainly use `r2`)
- `alpha` — LoRA scaling factor
- `n_experts` — number of experts (3: AlgoLogic, SyntaxPoly, SecureDebug)
- `BATCH_SIZE`, `MAX_LENGTHS` (dict with `"default"` and `"expert_0"` keys — expert 0 gets its own max length), `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`
- `SMOKE_TEST` — when `True`, caps samples per expert to 4 (train) / 3 (val) for a fast sanity run before a full one; also skips saving the end-of-epoch checkpoint

## 5. Running training

**Smoke test first (recommended):** set `SMOKE_TEST = True` in the script you're using and run it. This caps the dataset to a handful of samples per expert so you can catch shape/OOM/import errors in a couple of minutes before committing to a full run.

```bash
# via main.py (src/train.py, all experts combined, no val split)
python main.py

# via the val/logging-enabled script (what you've been using for MALORA)
python -m src.train_lora_moelora
```

Set `SMOKE_TEST = False` and re-run for the full training run.

### What happens during training
- Model loads in 4-bit (`nf4`, double quant, fp16 compute) via `bitsandbytes`, then `prepare_model_for_kbit_training` + `updating_layers()` swap in the chosen adapter.
- Optimizer is `bnb.optim.AdamW8bit` over only the trainable (adapter/router) parameters.
- Each step computes `task_loss` (LM cross-entropy) plus `aux_loss` (load-balancing loss collected from every `MALoRADownProjLayer`/`SymmetricMoEDownProjLayer` in the model) — weighted `0.01×` in `train.py`, unweighted (`×1`) in `train_lora_moelora.py`.
- Gradients are clipped to `max_norm=1.0`.
- NaNs in `task_loss`, `aux_loss`, or `total_loss` raise immediately rather than silently corrupting the run.
- Checkpoints (full `model.state_dict()`) save to `checkpoints/` every 1000 steps (`epoch{N}_step{M}.pt`) and at the end of each epoch (`epoch_{N}.pt`), unless `SMOKE_TEST=True`.
- `train_lora_moelora.py` additionally runs a validation pass at the end of every epoch and appends `{stage, epoch, val_task_loss, val_aux_loss}` to `results/model_train.jsonl`.

`checkpoints/` is gitignored — checkpoints won't be committed, so back them up separately (e.g. to your own storage).

## 6. Inference

`inference.py` gives a simple terminal Reply against a saved checkpoint:

```bash
python inference.py
```

Before running, make sure the hyperparameters (`r1`, `r2`, `alpha`, `n_experts`, `layer_range`) and `CHECKPOINT` path at the top of `inference.py` match what you actually trained with —  Type your prompt, then `END` on its own line to submit; type `exit` to quit.

## 7. Evaluation

`src/evaluate.py` and `Multipl.py` integrate with `lm-evaluation-harness` for HumanEval /MBPP/ MultiPL-E scoring, and `results/` contains prior benchmark outputs (`multipl_cpp_lora_full/`, `multipl_js_lora_full/`, etc.). 

