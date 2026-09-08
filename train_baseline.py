from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text

if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]
    test = dataset["test"]

    print("Cleaning text...")
    train_texts = [clean_text(t) for t in train["text"]]
    test_texts = [clean_text(t) for t in test["text"]]
    y_train = train["label"]
    y_test = test["label"]

    print("Vectorizing (fit on train only)...")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)  # transform only, no fit -- avoids leakage

    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_acc = accuracy_score(y_test, logreg_pred)

    print("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)

    print("\n--- Test accuracy comparison ---")
    print(f"Logistic Regression: {logreg_acc:.4f}")
    print(f"Naive Bayes:         {nb_acc:.4f}")

    print("\n--- Reviews where the two models disagree ---")
    label_names = {0: "negative", 1: "positive"}
    disagreements = [i for i in range(len(y_test)) if logreg_pred[i] != nb_pred[i]]
    print(f"Models disagree on {len(disagreements)} / {len(y_test)} reviews")
    for i in disagreements[:3]:
        print(f"\nReview {i} (true={label_names[y_test[i]]}, "
              f"logreg={label_names[logreg_pred[i]]}, nb={label_names[nb_pred[i]]}):")
        print(test[i]["text"][:150], "...")

    print("\n--- Logistic Regression: top words by learned weight ---")
    feature_names = vectorizer.get_feature_names_out()
    coefs = logreg.coef_[0]
    word_weights = list(zip(feature_names, coefs))
    word_weights.sort(key=lambda pair: pair[1])

    print("\nTop 15 most NEGATIVE-weighted words:")
    for word, weight in word_weights[:15]:
        print(f"{word:15s} {weight:.3f}")

    print("\nTop 15 most POSITIVE-weighted words:")
    for word, weight in word_weights[-15:][::-1]:
        print(f"{word:15s} {weight:.3f}")
