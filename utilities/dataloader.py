import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input = []
        self.target = []

        token = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token) - max_length, stride):
            input_chunk = token[i:i + max_length]
            target_chunk = token[i + 1: i + max_length + 1]
            self.input.append(torch.tensor(input_chunk))
            self.target.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset onject
    dataset = TextDataset(txt, tokenizer, max_length, stride)

    # Create dataloader object
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader

# Convinience functions for converting text to tokens and tokens to text
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())