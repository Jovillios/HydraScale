from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import torch


class HydraDataLoader(DataLoader):
    def __init__(self, seq_len: int, batch_size: int, dataset_name: str, tokenizer_name: str, num_workers: int = 1,  num_proc: int = 1, subset: int | None = None, split: str = "train") -> None:
        # parameters
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_proc = num_proc

        if subset:
            self.dataset = self.dataset.select(range(subset))

        self.tokenized_dataset = self.tokenize_dataset(self.dataset)
        super().__init__(
            self.tokenized_dataset,
            collate_fn=self.collate_batch,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=False,
        )

    def tokenize_group_text(self, examples, tokenize, seq_len):
        tokenized_text_batch = tokenize(
            examples["content"],
            return_attention_mask=False,
            return_tensors="np")
        
        concatenated_tokens = { "input_ids": np.concatenate(tokenized_text_batch["input_ids"], axis=0) }
        total_length = concatenated_tokens["input_ids"].shape[0]
        total_length = (total_length // seq_len) * seq_len  # Trim to multiple of seq_len
       
        result = {
            "input_ids": concatenated_tokens["input_ids"][:total_length].reshape(-1, seq_len)
        }
        print(result["input_ids"].shape)
        return result
    
    def tokenize_dataset(self, dataset):
        return dataset.map(
            lambda examples: self.tokenize_group_text(examples, self.tokenizer, self.seq_len+1),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_proc,
        )

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch], dim=0)
        batch_size = batch_input_ids.size(0)
        input_ids = batch_input_ids[: , :-1].contiguous()
        targets_ids = batch_input_ids[:, 1:].contiguous()

        position_ids = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()

        attn_mask = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


        return {
            "input_ids": input_ids,
            "targets": targets_ids,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "hidden_states": None,
        }
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise StopIteration
        return batch
       

if __name__ == "__main__":
    dataloader = HydraDataLoader(
        seq_len=128,
        batch_size=4,
        dataset_name="ProCreations/Ultra-FineWeb-EDU",
        tokenizer_name="HuggingFaceTB/SmolLM-360M-Instruct",
        num_workers=0,
        subset=1000,
        split="train"
    )

    print(len(dataloader))
    for batch in dataloader:
        print("Input IDs shape:", batch["input_ids"].shape)
        print("Targets shape:", batch["targets"].shape)
        print("Position IDs shape:", batch["position_ids"].shape)
        print("Attention Mask shape:", batch["attn_mask"].shape)
        break

