from matplotlib import pyplot as plt
import torch

from utilities.dataloader import create_dataloader
from train import plot_losses, train_model
from model.transformer import TransformerModel


def main(config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    with open(file_path, "r", encoding="utf-8") as file: text_data = file.read()

    # Initialize model
    model = TransformerModel(config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay=0.1)

    # Set up dataloaders
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=config["ctx_len"],
        stride=config["ctx_len"],
        drop_last=True,
        shuffle=True
    )

    val_loader = create_dataloader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=config["ctx_len"],
        stride=config["ctx_len"],
        drop_last=False,
        shuffle=False
    )

    # Train model
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device,
        n_epochs=settings["num_epochs"],
        eval_freq=5, eval_iter=1,
        start_context="Once upon a time,",
        warmup_steps=10, initial_lr=1e-5, min_lr=1e-5
    )

    return train_losses, val_losses, tokens_seen, model, optimizer


if __name__ == "__main__":

    CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "ctx_len": 1024,        # context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.0,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "num_epochs": 5,
        "batch_size": 3,
    }

    # Initiate training
    train_losses, val_losses, tokens_seen, model, optimizer = main(CONFIG, OTHER_SETTINGS)

    # After training
    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save model for future loading and inference
    torch.save(model.state_dict(), "model.pth")