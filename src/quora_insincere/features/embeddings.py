import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def extract_embeddings(zip_path: str | Path, extract_path: str | Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def load_embeddings(filepath: str | Path) -> dict[str, np.ndarray]:
    embeddings_index = {}
    fail_count = 0

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                values = line.rstrip().split(" ")
                if len(values) == 301:
                    word = values[0]
                    vector = np.asarray(values[1:], dtype="float32")
                else:
                    word = "".join(values[:-300])
                    vector = np.asarray(values[-300:], dtype="float32")
                embeddings_index[word] = vector
            except Exception:
                fail_count += 1

    print(f"Loaded {len(embeddings_index)} word vectors")
    print(f"Failed lines: {fail_count}")
    return embeddings_index


def check_coverage(vocab, embeddings_index: Mapping[str, np.ndarray]):
    known_words = {}
    unknown_words = {}
    known_count = 0
    unknown_count = 0

    for word in vocab.keys():
        if word in embeddings_index:
            known_words[word] = embeddings_index[word]
            known_count += vocab[word]
        else:
            unknown_words[word] = vocab[word]
            unknown_count += vocab[word]

    vocab_coverage = len(known_words) / len(vocab)
    text_coverage = known_count / (known_count + unknown_count)
    print(f"Found embeddings for {vocab_coverage:.2%} of vocab")
    print(f"Found embeddings for {text_coverage:.2%} of all text")
    return sorted(unknown_words.items(), key=lambda x: x[1], reverse=True)


def build_embedding_matrix(tokenizer, embeddings_index, max_vocab_size: int, embed_dim: int) -> np.ndarray:
    vocab_size = max_vocab_size + 1
    embedding_matrix = np.zeros((vocab_size, embed_dim))

    for word, i in tokenizer.word_index.items():
        if i >= max_vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

