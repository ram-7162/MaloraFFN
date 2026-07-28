r1   = 32
r2  = 96
alpha   = 32.0
n_experts  = 3
BATCH_SIZE  = 1
MAX_LENGTHS = {
    "default": 512,
    "expert_0": 512 
}

EPOCHS      = 3
MODE        = "malora"
SEED        = 42 
LEARNING_RATE = 1.32e-4
WEIGHT_DECAY = 0.065





import json
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader


def load_and_split(jsonl_paths, val_ratio=0.1, seed=42):
    train_samples, val_samples = [], []
    rng = random.Random(seed)

    for expert_id, path in jsonl_paths.items():
        rows = []
        with open(path, 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                row['expert_id'] = expert_id  
                rows.append(row)

        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio))
        val_samples.extend(rows[:n_val])
        train_samples.extend(rows[n_val:])

        print(f"Expert {expert_id}: {len(rows)} total -> {len(rows)-n_val} train / {n_val} val")

    print(f"Total: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples



class MALoRADataset(Dataset):

    def __init__(self, samples, tokenizer, max_lengths, min_output_tokens=5):
        self.tokenizer = tokenizer
        self.max_lengths = max_lengths

        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.samples = []
        dropped = 0

        original_by_expert = Counter(r['expert_id'] for r in samples) 
        dropped_by_expert = Counter()
        
        for row in samples:
            prompt_str = self._prompt_prefix(row)
            prompt_len = len(tokenizer(prompt_str, add_special_tokens=False)['input_ids'])
            
            expert_id = row.get('expert_id', -1)
            if expert_id == 0:
                current_max_len = self.max_lengths.get("expert_0", 512)
            else:
                current_max_len = self.max_lengths.get("default", 512)

            if prompt_len >= current_max_len - min_output_tokens:
                dropped_by_expert[row['expert_id']] += 1
                dropped += 1  
                continue
                
            row = dict(row)             
            row['_prompt_len'] = prompt_len
            self.samples.append(row)

        if dropped > 0:
            print(f"Dropped {dropped}/{len(samples)} samples — prompt too long for designated max_lengths")
            print(f"Dropped by expert: {dict(dropped_by_expert)}")


    def __len__(self):
        return len(self.samples)


    def _prompt_prefix(self, row):
        user_content = f"{row['instruction']}\n\nInput: {row['input']}".strip() if row.get('input') else row['instruction']
        
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def format_prompt(self, row):
        user_content = f"{row['instruction']}\n\nInput: {row['input']}".strip() if row.get('input') else row['instruction']
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row['output']}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

    
    def __getitem__(self, idx):
        row = self.samples[idx]
        full_text = self.format_prompt(row)

        expert_id = row.get('expert_id', -1)
        if expert_id == 0:
            current_max_len = self.max_lengths.get("expert_0", 512)
        else:
            current_max_len = self.max_lengths.get("default", 512)

        encoded = self.tokenizer(
            full_text, 
            max_length=current_max_len,
            truncation=True, 
            padding='max_length', 
            return_tensors='pt',
            add_special_tokens=False  # Chat template already added <|begin_of_text|>
        )

        prompt_len = min(row['_prompt_len'], current_max_len)

        input_ids      = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'expert_id':      torch.tensor(row['expert_id'], dtype=torch.long)
        }


def get_dataloaders(jsonl_paths, tokenizer, batch_size=4, max_length=MAX_LENGTHS,
                     val_ratio=0.1, seed=SEED, samples_per_expert=None):

    train_samples, val_samples = load_and_split(jsonl_paths, val_ratio, seed)
    

    if samples_per_expert is not None:
        capped_train = {}
        for row in train_samples:
            capped_train.setdefault(row['expert_id'], []).append(row)
        train_samples = [r for rows in capped_train.values() for r in rows[:samples_per_expert]]

        capped_val = {}
        for row in val_samples:
            capped_val.setdefault(row['expert_id'], []).append(row)
        val_samples = [r for rows in capped_val.values() for r in rows[:3]]

    train_ds = MALoRADataset(train_samples, tokenizer, max_length)
    val_ds   = MALoRADataset(val_samples, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows



def get_dataloaders2(split_path, tokenizer, batch_size=4, max_length=512, shuffle=True):
    samples = load_jsonl(split_path)
    ds = MALoRADataset(samples, tokenizer, max_length)
    
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

