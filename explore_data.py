from datasets import load_dataset

# Load the IMDB movie review dataset (50k reviews, binary sentiment: 0=neg, 1=pos)
dataset = load_dataset("stanfordnlp/imdb")

print("Dataset splits:", dataset)

train = dataset["train"]
print("\nColumns:", train.column_names)
print("Number of training examples:", len(train))

print("\n--- A few raw examples ---")
for i in [0, 1, 12500]:
    example = train[i]
    print(f"\nExample {i} (label={example['label']}):")
    print(example["text"][:300], "...")

print("\n--- Label distribution (train) ---")
labels = train["label"]
print("0 (negative):", labels.count(0))
print("1 (positive):", labels.count(1))

print("\n--- Review length (train, word count) ---")
word_counts = [len(text.split()) for text in train["text"]]
print("min:", min(word_counts))
print("max:", max(word_counts))
print("mean:", sum(word_counts) / len(word_counts))
sorted_counts = sorted(word_counts)
print("median:", sorted_counts[len(sorted_counts) // 2])

print("\n--- '<br />' tag check (train) ---")
num_with_br = sum(1 for text in train["text"] if "<br />" in text)
print(f"{num_with_br} / {len(train)} reviews contain '<br />'")
print("Example snippet with tags:")
for text in train["text"]:
    if "<br />" in text:
        idx = text.index("<br />")
        print(text[max(0, idx - 60):idx + 60])
        break
