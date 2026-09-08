import torch
import torch.nn as nn

from datasets import load_dataset
from preprocess import clean_text
from vocab import build_vocab, encode_and_pad


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len) integer word indices, 0 = <PAD>
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq_len, 1): 1 for real tokens, 0 for pad

        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        summed = (embedded * mask).sum(dim=1)  # zero out pad positions, sum over sequence
        counts = mask.sum(dim=1).clamp(min=1)  # number of real tokens per review (avoid div by 0)
        pooled = summed / counts  # (batch, embedding_dim): masked mean pooling

        hidden = self.relu(self.fc1(pooled))
        logit = self.fc2(hidden).squeeze(-1)  # (batch,)
        return logit


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]

    print("Preparing a small batch for a smoke test...")
    train_texts = [clean_text(t) for t in train["text"][:100]]  # small slice, just for vocab+batch demo
    word2idx, _ = build_vocab(train_texts, max_vocab_size=20000)

    MAX_LEN = 256
    batch_texts = train_texts[:8]
    batch = [encode_and_pad(t, word2idx, MAX_LEN) for t in batch_texts]
    batch_tensor = torch.tensor(batch, dtype=torch.long)
    print("Input batch shape:", batch_tensor.shape, "(batch_size x max_len)")

    model = SentimentNet(vocab_size=len(word2idx))
    num_params = sum(p.numel() for p in model.parameters())
    print("Model parameter count:", num_params)

    logits = model(batch_tensor)
    print("Output logits shape:", logits.shape, "(one raw score per review)")
    print("Sample output logits:", logits.detach().numpy())
