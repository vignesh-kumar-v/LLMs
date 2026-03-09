from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories")

with open("train.txt", "w", encoding="utf-8") as f:
    for example in dataset["train"]:
        f.write(example["text"] + "\n")

with open("val.txt", "w", encoding="utf-8") as f:
    for example in dataset["validation"]:
        f.write(example["text"] + "\n")

print("Saved train.txt and val.txt")
