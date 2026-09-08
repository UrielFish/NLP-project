from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import clean_text, remove_stopwords


def train_and_evaluate(train_texts, test_texts, y_train, y_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]
    test = dataset["test"]
    y_train = train["label"]
    y_test = test["label"]

    print("Cleaning text...")
    train_cleaned = [clean_text(t) for t in train["text"]]
    test_cleaned = [clean_text(t) for t in test["text"]]

    print("Removing stopwords (negation words kept)...")
    train_no_sw = [remove_stopwords(t) for t in train_cleaned]
    test_no_sw = [remove_stopwords(t) for t in test_cleaned]

    print("Training Logistic Regression WITH stopwords...")
    acc_with_sw = train_and_evaluate(train_cleaned, test_cleaned, y_train, y_test)

    print("Training Logistic Regression WITHOUT stopwords...")
    acc_no_sw = train_and_evaluate(train_no_sw, test_no_sw, y_train, y_test)

    print("\n--- Test accuracy comparison ---")
    print(f"With stopwords:    {acc_with_sw:.4f}")
    print(f"Without stopwords: {acc_no_sw:.4f}")
    print(f"Difference:        {acc_no_sw - acc_with_sw:+.4f}")
