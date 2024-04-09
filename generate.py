import tiktoken
import torch

from dataloader import text_to_token_ids, token_ids_to_text
from transformer import TransformerModel


def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "ctx_len": 1024,      # Context length
    "emb_dim": 768,       # Embedding dimension
    "n_heads": 12,        # Number of attention heads
    "n_layers": 12,       # Number of layers
    "drop_rate": 0,       # Dropout rate
    "qkv_bias": False     # Query-Key-Value bias
}

torch.manual_seed(123)
model = TransformerModel(CONFIG)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # disable dropout

tokenizer = tiktoken.get_encoding("gpt2")

start_context = """Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day,"""

encoded = text_to_token_ids(start_context, tokenizer)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print("\nInput text:", start_context)

out = generate(
    model=model,
    idx=encoded,
    max_new_tokens=10,
    context_size=CONFIG["ctx_len"],
    top_k=10,
    temperature=1
)

decoded_text = token_ids_to_text(out, tokenizer)

print("Output length:", len(out[0]))
print("Output text:", decoded_text)