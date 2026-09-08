from collections import Counter

from datasets import load_dataset

from preprocess import clean_text

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(texts, max_vocab_size=20000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    most_common = counter.most_common(max_vocab_size)

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _count in most_common:
        word2idx[word] = len(word2idx)

    return word2idx, counter


def encode(text, word2idx):
    unk_idx = word2idx[UNK_TOKEN]
    return [word2idx.get(word, unk_idx) for word in text.split()]


def pad_or_truncate(encoded, max_len):
    if len(encoded) >= max_len:
        return encoded[:max_len]
    return encoded + [0] * (max_len - len(encoded))  # 0 = <PAD>


def encode_and_pad(text, word2idx, max_len):
    return pad_or_truncate(encode(text, word2idx), max_len)


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"]

    print("Cleaning text...")
    train_texts = [clean_text(t) for t in train["text"]]

    print("Building vocabulary...")
    word2idx, counter = build_vocab(train_texts, max_vocab_size=20000)

    print("\n--- Vocab summary ---")
    print("Vocab size (incl. PAD/UNK):", len(word2idx))
    print("Total distinct words seen:", len(counter))

    print("\n--- 10 most common words ---")
    for word, count in counter.most_common(10):
        print(f"{word:15s} {count}")

    print("\n--- Example encoding ---")
    for i in [0, 1]:
        text = train_texts[i]
        encoded = encode(text, word2idx)
        num_unk = sum(1 for idx in encoded if idx == word2idx[UNK_TOKEN])
        print(f"\nReview {i}: {len(encoded)} tokens, {num_unk} are <UNK> "
              f"({num_unk / len(encoded):.1%})")
        print("First 15 words:", text.split()[:15])
        print("First 15 indices:", encoded[:15])

    print("\n--- Padding/truncation demo (max_len=256) ---")
    MAX_LEN = 256
    short_review = train_texts[0]  # 289 tokens -> will be truncated
    padded = encode_and_pad(short_review, word2idx, MAX_LEN)
    print(f"Review 0: original {len(short_review.split())} tokens -> padded length {len(padded)}")
    print("Last 10 values (should be real words, this one gets truncated not padded):", padded[-10:])

    # find a genuinely short review to show padding with zeros
    for i, text in enumerate(train_texts):
        if len(text.split()) < 20:
            padded_short = encode_and_pad(text, word2idx, MAX_LEN)
            print(f"\nReview {i}: original {len(text.split())} tokens -> padded length {len(padded_short)}")
            print("Last 10 values (should be 0s, i.e. <PAD>):", padded_short[-10:])
            break
