"""Bigram model for text generation"""

import itertools
from typing import Iterator

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def get_chars() -> str:
    """Returns 'abcdefghijklmnopqrstuvwxyz'."""
    return "".join([chr(i) for i in range(ord("a"), ord("z") + 1)])


def get_itos_stoi() -> tuple[dict[int, str], dict[str, int]]:
    """Returns integer to character and character to integer mappings."""
    chars = get_chars()
    itos: dict[int, str] = {0: "."}
    itos.update({i + 1: s for i, s in enumerate(chars)})
    stoi: dict[str, int] = {s: i for i, s in itos.items()}
    return itos, stoi


def get_bigram_counts(words: list[str]) -> Tensor:
    """Gets the number of occurrences of each bigram in a list of words."""
    counts: Tensor = torch.zeros((27, 27), dtype=torch.int32)
    _, stoi = get_itos_stoi()
    for w in words:
        for ch1, ch2 in get_bigram_iterator(w):
            idx1: int = stoi[ch1]
            idx2: int = stoi[ch2]
            counts[idx1, idx2] += 1
    return counts


def train_bigram_counting(training_set: list[str]) -> Tensor:
    """Trains a bigram model on a list of words and returns the probability matrix."""
    counts = get_bigram_counts(training_set).float()
    return counts / counts.sum(dim=1, keepdim=True)


def plot_bigram(counts: Tensor) -> None:
    """
    Plots a 27x27 matrix of bigram counts. The first row and column are
    the empty string, ".", and the remaining rows and columns are the
    26 letters of the alphabet. The counts are displayed in the cells.
    """
    itos, _ = get_itos_stoi()

    plt.figure(figsize=(16, 16))
    plt.imshow(counts, cmap="Blues")

    def get_kwargs(i, j) -> dict[str, any]:
        return {"x": j, "y": i, "ha": "center", "color": "gray"}

    for i, j in itertools.product(range(27), repeat=2):
        kwargs: dict[str, any] = get_kwargs(i, j)
        plt.text(s=itos[i] + itos[j], va="bottom", **kwargs)
        plt.text(s=counts[i, j].item(), va="top", **kwargs)
    plt.axis("off")
    plt.savefig("images/bigram.png")


def get_bigram_iterator(word: str) -> Iterator[tuple[str, str]]:
    """Returns an iterator over the bigrams of a word."""
    chs: list[str] = ["."] + list(word) + ["."]
    return zip(chs, chs[1:])


def loss(word: str, probabilities: Tensor) -> Tensor:
    """Computes the negative log-likelihood of a word under a bigram model."""
    _, stoi = get_itos_stoi()

    retval = torch.tensor(0.0)
    count = 0
    for ch1, ch2 in get_bigram_iterator(word):
        idx1: int = stoi[ch1]
        idx2: int = stoi[ch2]
        retval += -probabilities[idx1, idx2].log()
        count += 1

    return retval / count


def sample_words(probabilities: Tensor, n: int = 1, _rng=None) -> str | list[str]:
    """
    Given a bigram probability matrix, samples a word from the distribution if
    n = 1, or a list of n words if n > 1.
    """
    itos, _ = get_itos_stoi()

    _rng = _rng or torch.Generator().manual_seed(0x2024_03_19)

    words: list[str] = []
    for _ in range(n):
        idx = 0
        out: list[str] = []
        while True:
            p: Tensor = probabilities[idx]
            idx: int = torch.multinomial(
                p, num_samples=1, replacement=True, generator=_rng
            ).item()
            out.append(itos[idx])
            if idx == 0:
                break
        word = "".join(out)
        if n == 1:
            return word
        words.append(word)
    return words


def eval_probability_matrix(probabilities: Tensor, validation_set: list[str]) -> Tensor:
    """Computes the average negative log likelihood of each word in the training set."""
    total_loss: Tensor = torch.tensor(
        list(map(lambda w: float(loss(w, probabilities)), validation_set))
    )
    return total_loss.mean()
