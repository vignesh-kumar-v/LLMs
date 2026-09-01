"""Download TinyStories and write it out as plain text.

Two changes from the original:

* Documents are separated by an explicit ``<|endoftext|>`` marker rather than a
  bare newline. Without a boundary token the model is trained to run one story
  straight into the next, and has no way to learn where a story ends — which is
  exactly what makes generated samples ramble past their natural ending.
* Writes are buffered through an iterator instead of indexing the dataset row
  by row, and the split names are handled in one loop.

Run once:  python TinyStories.py
"""

import os

from datasets import load_dataset

EOT = "<|endoftext|>"


def write_split(dataset_split, path):
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            text = example["text"].strip()
            if not text:
                continue
            f.write(text)
            f.write("\n")
            f.write(EOT)
            f.write("\n")
            count += 1
    size_mb = os.path.getsize(path) / 1024**2
    print(f"Wrote {count:,} stories to {path} ({size_mb:.1f} MB)")


def main():
    print("Downloading roneneldan/TinyStories ...")
    dataset = load_dataset("roneneldan/TinyStories")
    write_split(dataset["train"], "train.txt")
    write_split(dataset["validation"], "val.txt")
    print("Done. Next: python main.py")


if __name__ == "__main__":
    main()
