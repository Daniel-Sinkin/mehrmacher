"""Example showing the bigram functionality using simple counting probabilities."""

import torch
from torch import Tensor

from src.bigram import (
    eval_probability_matrix,
    get_bigram_counts,
    loss,
    plot_bigram,
    sample_words,
)
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

    print("Samling words (higher loss means less likely)")
    print(f" idx | {'generated word':25}| {'  loss'}")
    print("-----+" + (25 + 1) * "-" + "+--------")
    for i in range(20):
        word = sample_words(probabilities, _rng=_rng)
        _loss = loss(word, probabilities)
        print(f"{i:4} | {word:25}| {_loss.item():7.3f}")

    print()

    word_n_losses: list[tuple[str, float]] = [
        (word, loss(word, probabilities).item()) for word in names
    ]
    word_n_losses.sort(key=lambda x: x[1])
    print("Most likely words")
    for word, loss_ in word_n_losses[:5]:
        print(f"\t{word:25} {loss_:4.2f}")

    print("Least likely words")
    for word, loss_ in word_n_losses[-5:]:
        print(f"\t{word:25} {loss_:4.2f}")

    print(
        f"Average loss over the list of words: {eval_probability_matrix(probabilities, names).item():.2f}"
    )


if __name__ == "__main__":
    main()
