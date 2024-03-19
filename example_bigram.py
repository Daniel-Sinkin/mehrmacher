"""Example showing the bigram functionality."""

import torch
from torch import Tensor

from src.bigram import get_bigram_counts, loss, plot_bigram, sample_word
from src.util import get_names_list

ITERATION_LIMIT = 100
_rng = torch.Generator().manual_seed(2147483647)


def main():
    """Example showing the bigram functionality."""
    names = get_names_list()
    counts = get_bigram_counts(names)

    plot_bigram(counts)

    probabilities: Tensor = (counts + 1).float()
    probabilities: Tensor = probabilities / probabilities.sum(dim=1, keepdim=True)

    print(f" idx | {'generated word':40}| {'  loss'}")
    print("-----+" + (41) * "-" + "+--------")
    for i in range(20):
        word = sample_word(probabilities, _rng)
        _loss = loss(word, probabilities)
        print(f"{i:4} | {word:40}| {_loss.item():7.3f}")


if __name__ == "__main__":
    main()
