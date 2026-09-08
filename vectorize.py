from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_text, remove_stopwords


def top_words(X_train, vectorizer, row_index, n=10):
    row = X_train[row_index]
    feature_names = vectorizer.get_feature_names_out()
    word_scores = list(zip(row.indices, row.data))
    word_scores.sort(key=lambda pair: pair[1], reverse=True)
    return [(feature_names[idx], score) for idx, score in word_scores[:n]]


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]

    print("Cleaning text...")
    cleaned_texts = [clean_text(t) for t in train["text"]]
    no_stopword_texts = [remove_stopwords(t) for t in cleaned_texts]

    print("Fitting TF-IDF vectorizer (with stopwords)...")
    vectorizer_with_sw = TfidfVectorizer()
    X_with_sw = vectorizer_with_sw.fit_transform(cleaned_texts)

    print("Fitting TF-IDF vectorizer (stopwords removed)...")
    vectorizer_no_sw = TfidfVectorizer()
    X_no_sw = vectorizer_no_sw.fit_transform(no_stopword_texts)

    print("\n--- Vocabulary size comparison ---")
    print("With stopwords:   ", len(vectorizer_with_sw.vocabulary_))
    print("Without stopwords:", len(vectorizer_no_sw.vocabulary_))

    print("\n--- Top 10 TF-IDF words for review 0 ---")
    print(f"{'WITH stopwords':25s} | {'WITHOUT stopwords':25s}")
    with_sw = top_words(X_with_sw, vectorizer_with_sw, 0)
    no_sw = top_words(X_no_sw, vectorizer_no_sw, 0)
    for (w1, s1), (w2, s2) in zip(with_sw, no_sw):
        print(f"{w1:15s} {s1:.4f}     | {w2:15s} {s2:.4f}")
