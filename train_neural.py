import json

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import SentimentNet
from preprocess import clean_text
from vocab import build_vocab, encode_and_pad

MAX_LEN = 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1
PATIENCE = 2


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return total_loss / total, correct / total


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]

    print("Cleaning text...")
    all_texts = [clean_text(t) for t in train["text"]]
    all_labels = train["label"]

    print(f"Splitting into train/validation ({1 - VAL_FRACTION:.0%}/{VAL_FRACTION:.0%}, stratified)...")
    train_texts, val_texts, y_train, y_val = train_test_split(
        all_texts, all_labels, test_size=VAL_FRACTION, stratify=all_labels, random_state=42
    )
    print(f"Train: {len(train_texts)} reviews, Validation: {len(val_texts)} reviews")

    print("Building vocabulary (from training subset only)...")
    word2idx, _ = build_vocab(train_texts, max_vocab_size=20000)

    print("Encoding and padding...")
    X_train = [encode_and_pad(t, word2idx, MAX_LEN) for t in train_texts]
    X_val = [encode_and_pad(t, word2idx, MAX_LEN) for t in val_texts]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentNet(vocab_size=len(word2idx), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining for up to {NUM_EPOCHS} epochs ({len(train_loader)} batches/epoch), "
          f"early stopping patience={PATIENCE}...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8}")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        train_loss = total_loss / len(train_dataset)

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = "  <- best so far"
        else:
            epochs_without_improvement += 1

        print(f"{epoch + 1:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_acc:>8.4f}{marker}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nStopping early: validation loss hasn't improved for {PATIENCE} epochs.")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "sentiment_net.pt")
    with open("word2idx.json", "w") as f:
        json.dump(word2idx, f)
    print(f"\nRestored best checkpoint (val loss {best_val_loss:.4f}) and saved to sentiment_net.pt")
