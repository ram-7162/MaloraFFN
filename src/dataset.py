import json
import torch
from torch.utils.data import Dataset, DataLoader

class MALoRADataset(Dataset):

    def __init__(self, jsonl_paths, tokenizer, max_length=512, samples_per_expert=None):
        """
        jsonl_paths: dict like {0: 'data/expert0_algo_training.jsonl', ...}
        samples_per_expert: int or None — if int, takes only that many per file (for smoke test)
        """
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.samples     = []

        for expert_id, path in jsonl_paths.items():
            rows = []
            with open(path, 'r') as f:
                for line in f:
                    rows.append(json.loads(line.strip()))

            # Take only N samples if specified (smoke test mode)
            if samples_per_expert is not None:
                rows = rows[:samples_per_expert]

            self.samples.extend(rows)
            print(f"Expert {expert_id} loaded: {len(rows)} samples from {path}")

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def format_prompt(self, row):
        inp = row.get('input', '').strip()
        if inp:
            return (
                f"### Instruction:\n{row['instruction']}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Output:\n{row['output']}"
            )
        else:
            return (
                f"### Instruction:\n{row['instruction']}\n\n"
                f"### Output:\n{row['output']}"
            )

    def __getitem__(self, idx):
        row    = self.samples[idx]
        prompt = self.format_prompt(row)

        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids':      encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'expert_id':      torch.tensor(row['expert_id'], dtype=torch.long)
        }


def get_dataloader(jsonl_paths, tokenizer, batch_size=4,
                   max_length=512, samples_per_expert=None, shuffle=True):

    dataset = MALoRADataset(
        jsonl_paths,
        tokenizer,
        max_length=max_length,
        samples_per_expert=samples_per_expert
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
