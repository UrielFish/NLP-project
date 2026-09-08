import re

from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negation words we deliberately keep even though they're in the standard
# stopword list -- removing them would flip sentiment, e.g. "not good" -> "good"
NEGATION_WORDS = {"not", "no", "nor", "never", "cannot", "none", "nobody", "nothing", "without", "noone"}
STOP_WORDS = ENGLISH_STOP_WORDS - NEGATION_WORDS


def clean_text(text: str) -> str:
    text = text.replace("<br />", " ")
    text = text.lower()
    text = text.replace("'", "")  # don't -> dont (delete, don't split into two words)
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # other punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()  # collapse repeated whitespace
    return text


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]

    print("--- Before / after cleaning (a few examples) ---")
    for i in [0, 1, 12500]:
        raw = train[i]["text"]
        cleaned = clean_text(raw)
        print(f"\nExample {i} BEFORE:")
        print(raw[:300])
        print(f"\nExample {i} AFTER:")
        print(cleaned[:300])

    print("\n--- Stopword removal check ---")
    test_phrase = "this movie is not good and it was never funny either"
    print("BEFORE:", test_phrase)
    print("AFTER: ", remove_stopwords(test_phrase))

    print("\n--- Stopword removal on example 1 ---")
    cleaned_1 = clean_text(train[1]["text"])
    no_stop_1 = remove_stopwords(cleaned_1)
    print("BEFORE:", cleaned_1[:300])
    print("AFTER: ", no_stop_1[:300])
