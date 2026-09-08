import json

import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import DataLoader, TensorDataset

from model import SentimentNet
from model_lstm import SentimentLSTM
from preprocess import clean_text
from vocab import encode_and_pad

MAX_LEN = 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 64


def report(name, y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    acc = (y_pred == y_true).mean() if hasattr(y_pred, "mean") else sum(
        p == t for p, t in zip(y_pred, y_true)
    ) / len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}  (of predicted positive, how many were right)")
    print(f"  Recall:    {recall:.4f}  (of actual positive, how many were caught)")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    return {"name": name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]
    test = dataset["test"]
    y_train = train["label"]
    y_test = test["label"]

    print("Cleaning text...")
    train_texts = [clean_text(t) for t in train["text"]]
    test_texts = [clean_text(t) for t in test["text"]]

    results = []

    # --- Logistic Regression + Naive Bayes (retrain quickly on TF-IDF) ---
    print("Vectorizing for TF-IDF models...")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    results.append(report("Logistic Regression (TF-IDF)", y_test, logreg.predict(X_test)))

    print("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    results.append(report("Naive Bayes (TF-IDF)", y_test, nb.predict(X_test)))

    # --- Neural net (mean pooling) ---
    print("\nLoading neural net (mean pooling)...")
    with open("word2idx.json") as f:
        word2idx_nn = json.load(f)
    nn_model = SentimentNet(vocab_size=len(word2idx_nn), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    nn_model.load_state_dict(torch.load("sentiment_net.pt"))
    nn_model.eval()

    X_test_nn = [encode_and_pad(t, word2idx_nn, MAX_LEN) for t in test_texts]
    nn_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_nn, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    nn_preds = []
    with torch.no_grad():
        for batch_x, _ in nn_loader:
            logits = nn_model(batch_x)
            nn_preds.extend((torch.sigmoid(logits) > 0.5).int().tolist())
    results.append(report("Neural net (mean pooling)", list(y_test), nn_preds))

    # --- LSTM ---
    print("\nLoading LSTM...")
    with open("word2idx_lstm.json") as f:
        word2idx_lstm = json.load(f)
    lstm_model = SentimentLSTM(vocab_size=len(word2idx_lstm), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    lstm_model.load_state_dict(torch.load("sentiment_lstm.pt"))
    lstm_model.eval()

    X_test_lstm = [encode_and_pad(t, word2idx_lstm, MAX_LEN) for t in test_texts]
    lstm_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_lstm, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    lstm_preds = []
    with torch.no_grad():
        for batch_x, _ in lstm_loader:
            logits = lstm_model(batch_x)
            lstm_preds.extend((torch.sigmoid(logits) > 0.5).int().tolist())
    results.append(report("LSTM (order-aware)", list(y_test), lstm_preds))

    # --- Summary table ---
    print("\n\n--- Summary ---")
    print(f"{'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for r in results:
        print(f"{r['name']:<30} {r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")
