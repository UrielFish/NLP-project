import torch
import torch.nn as nn

from datasets import load_dataset
from preprocess import clean_text
from vocab import build_vocab, encode_and_pad


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len) integer word indices, 0 = <PAD>
        lengths = (x != 0).sum(dim=1)  # true length of each review, before padding

        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _cell) = self.lstm(packed)  # hidden: (1, batch, hidden_dim) -- final step only
        final_hidden = hidden[-1]  # (batch, hidden_dim)

        logit = self.fc(final_hidden).squeeze(-1)  # (batch,)
        return logit


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]

    print("Preparing a small batch for a smoke test...")
    train_texts = [clean_text(t) for t in train["text"][:100]]
    word2idx, _ = build_vocab(train_texts, max_vocab_size=20000)

    MAX_LEN = 256
    batch_texts = train_texts[:8]
    batch = [encode_and_pad(t, word2idx, MAX_LEN) for t in batch_texts]
    batch_tensor = torch.tensor(batch, dtype=torch.long)
    print("Input batch shape:", batch_tensor.shape)

    model = SentimentLSTM(vocab_size=len(word2idx))
    num_params = sum(p.numel() for p in model.parameters())
    print("Model parameter count:", num_params)

    logits = model(batch_tensor)
    print("Output logits shape:", logits.shape, "(one raw score per review)")
    print("Sample output logits:", logits.detach().numpy())
