# NLP Sentiment Classification

A binary sentiment classifier for movie reviews, built as a CV/internship
project to demonstrate an end-to-end NLP workflow: raw text -> preprocessing
-> vectorization -> classical and neural models -> honest evaluation.

## Dataset

[IMDB movie reviews](https://huggingface.co/datasets/stanfordnlp/imdb) —
50,000 reviews, balanced 25k/25k train-test split, labeled positive (1) or
negative (0). Loaded via Hugging Face `datasets`.

## Methodology

**Preprocessing** (`preprocess.py`): lowercase, strip `<br />` tags left over
from the original HTML source, remove punctuation, collapse whitespace.
Stopword removal is available but deliberately keeps negation words
(`not`, `no`, `never`, ...) — removing them would flip meaning, e.g.
"not good" becoming "good". `stopword_comparison.py` and `vectorize.py`
compare vocabulary size and top TF-IDF terms with and without stopwords.

**Vectorization** — two representations, used by different models:
- **TF-IDF** (`sklearn.TfidfVectorizer`): sparse bag-of-words weighted by
  term frequency / inverse document frequency. Feeds the classical models.
- **Integer-encoded vocabulary** (`vocab.py`): a word-to-index mapping built
  from the training set (top 20k words, plus `<PAD>`/`<UNK>`), with reviews
  padded/truncated to a fixed length of 256 tokens. Feeds the neural models,
  which then learn their own embeddings from this.

**Models**:
- **Logistic Regression** and **Naive Bayes** (`train_baseline.py`) on
  TF-IDF features — interpretable classical baselines.
- **Feedforward neural net** (`model.py`, `train_neural.py`): learned word
  embeddings, mean-pooled across the sequence, then a small MLP head. Ignores
  word order.
- **LSTM** (`model_lstm.py`, `train_lstm.py`): learned embeddings fed through
  an LSTM, using the final hidden state as the review representation. Order-
  aware, unlike the other three.

All neural models are evaluated together in `evaluate_all.py`.

## Results

Evaluated on the 25,000-review IMDB test set.

| Model                          | Accuracy | Precision | Recall | F1    |
|---------------------------------|----------|-----------|--------|-------|
| Logistic Regression (TF-IDF)   | 88.3%    | 88.4%     | 88.2%  | 88.3% |
| Naive Bayes (TF-IDF)           | 83.0%    | 87.5%     | 77.1%  | 81.9% |
| Neural net (mean pooling)      | 85.6%    | 86.2%     | 84.9%  | 85.5% |
| LSTM (order-aware)             | 82.9%    | 82.3%     | 83.7%  | 83.0% |

Precision, recall, and F1 are reported alongside accuracy because they
surface trade-offs accuracy alone hides — e.g. Naive Bayes has high precision
but noticeably lower recall (77.1%), meaning it's conservative about calling
a review positive and misses more true positives than the other models.

## Discussion & limitations

The simplest model, Logistic Regression on TF-IDF, won outright. The LSTM —
despite being the only model that sees word order — performed *worse* than
both the logistic regression baseline and the order-blind mean-pooling
neural net. This is a genuine, if unglamorous, finding rather than a bug:

- IMDB reviews are long (median length well over 100 words), and sentiment
  signal is often carried by scattered opinion words ("boring", "brilliant")
  rather than sequences — exactly the setting where bag-of-words approaches
  are known to be competitive with, and sometimes beat, sequence models.
  See e.g. Wang & Manning (2012) on the strength of simple TF-IDF/NB-SVM
  baselines for sentiment tasks.
- The LSTM here uses randomly-initialized embeddings trained from scratch on
  only 25k reviews, no pretrained word vectors, and no regularization tuning
  — all of which make it prone to underfitting/overfitting relative to a
  convex, well-regularized logistic regression model.
- No hyperparameter search was done for any model; all neural models use a
  single fixed configuration (embedding dim 100, hidden dim 64, max length
  256 tokens).

**Not attempted, and worth calling out honestly:**
- Pretrained embeddings (GloVe/word2vec) or pretrained transformers (BERT),
  which would likely close or reverse the gap with the LSTM.
- Hyperparameter tuning / cross-validation for any model.
- Error analysis on which specific reviews each model gets wrong.

## Repository structure

```
explore_data.py          # first look at the dataset: shape, label balance, review lengths
preprocess.py             # text cleaning + stopword removal (keeps negation words)
vectorize.py               # TF-IDF vectorization + with/without-stopwords comparison
stopword_comparison.py    # accuracy impact of stopword removal (classical baseline)
vocab.py                   # integer vocabulary + padding/truncation for neural models
train_baseline.py          # Logistic Regression baseline (TF-IDF)
model.py / train_neural.py       # feedforward neural net (mean-pooled embeddings)
model_lstm.py / train_lstm.py    # LSTM (order-aware)
evaluate_neural.py, evaluate_lstm.py   # per-model evaluation
evaluate_all.py             # evaluates all four models together, prints summary table
requirements.txt
```

Trained model weights (`*.pt`) and vocabulary files (`word2idx*.json`) are
not committed — they're reproducible by re-running the corresponding
training script.

## How to run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python explore_data.py         # inspect the raw dataset
python train_baseline.py       # classical baseline (Logistic Regression)
python train_neural.py         # feedforward neural net -> sentiment_net.pt
python train_lstm.py           # LSTM -> sentiment_lstm.pt
python evaluate_all.py         # full comparison table (as reported above)
```
