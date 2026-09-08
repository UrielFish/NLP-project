import json

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

from model_lstm import SentimentLSTM
from preprocess import clean_text
from vocab import encode_and_pad

MAX_LEN = 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 64

if __name__ == "__main__":
    with open("word2idx_lstm.json") as f:
        word2idx = json.load(f)

    model = SentimentLSTM(vocab_size=len(word2idx), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("sentiment_lstm.pt"))
    model.eval()

    dataset = load_dataset("stanfordnlp/imdb")
    test = dataset["test"]

    print("Cleaning and encoding test set...")
    test_texts = [clean_text(t) for t in test["text"]]
    y_test = test["label"]
    X_test = [encode_and_pad(t, word2idx, MAX_LEN) for t in test_texts]

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total
    print(f"\nLSTM test accuracy: {acc:.4f}")

    print("\n--- Full comparison ---")
    print(f"Logistic Regression:        0.8831")
    print(f"Naive Bayes:                0.8301")
    print(f"Neural net (mean pooling):  0.8564")
    print(f"LSTM (order-aware):         {acc:.4f}")
