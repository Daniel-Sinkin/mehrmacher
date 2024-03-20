import random
from dataclasses import dataclass
from typing import Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.util import get_chars, get_itos_stoi, get_names_list


@dataclass
class Hyperparameters:
    EMBEDDING_SIZE = 2
    CHARSET_SIZE = 27
    CONTEXT_LENGTH = 3
    NUM_HIDDEN_NEURONS = 200
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 1000
    BATCH_SIZE = 32


class DatasetSplit(TypedDict):
    """
    Train, Develop, Test split, should only train on `train`, adjust
    hyperparameters on `dev` and evaluate on `test`. Number of runs on `test`
    should be limited to avoid overfitting.
    """

    train: tuple[Tensor, Tensor]
    dev: tuple[Tensor, Tensor]
    test: tuple[Tensor, Tensor]


def plot_learning_rate_finder(
    losses: Tensor, lrs, window_size=50, filepath=None
) -> float:
    """
    Get optimal learning rate for minimizing smoothed loss.
    """
    moving_average = np.convolve(
        losses, np.ones(window_size) / window_size, mode="valid"
    )

    plt.loglog(lrs, losses)
    plt.loglog(lrs[window_size - 1 :], moving_average)

    min_moving_avg = moving_average.min()
    min_moving_avg_arg = np.argmin(moving_average)
    plt.scatter(
        lrs[min_moving_avg_arg],
        min_moving_avg,
        color="r",
        zorder=5,
        label=f"Min moving average: {min_moving_avg:.2f} at lr = {lrs[min_moving_avg_arg]:.2f}",
    )

    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs Loss (log-log)\n smooting window of 50.")

    plt.legend()
    if filepath is not None:
        plt.savefig(filepath)


def build_dataset(words, hyperparams: Hyperparameters) -> tuple[Tensor, Tensor]:
    """
    Takes in words and hyperparameters to build a dataset, i.e. X as context
    and Y as the target.
    """
    _, stoi = get_itos_stoi()
    X, Y = [], []
    for w in words:
        context: list[int] = [0] * Hyperparameters.CONTEXT_LENGTH
        for ch in w + ".":
            idx: int = stoi[ch]
            X.append(context)
            Y.append(idx)
            context: list[int] = context[1:] + [idx]  # 1234, idx = 5 -> 2345

    # TODO: Check what dtype those two (should) have
    X: Tensor = torch.tensor(X)
    Y: Tensor = torch.tensor(Y)
    return X, Y


def build_datset_split(
    words,
    hyperparams: Hyperparameters,
    partition_pct: list[float] = None,
    seed: int = None,
) -> DatasetSplit:
    """
    Takes in a word, hyperparams and an optional seed to generate a split
    of the dataset into train, dev and test sets.

    The partition_pct is a list of two values that represent the percentage of
    the dataset that should be used for training and testing.

    Default is 80% training, 10% dev and 10% test.
    """
    if seed is not None:
        _rng = np.random.default_rng(seed)
    else:
        _rng = np.random.default_rng()

    if partition_pct is not None:
        assert len(partition_pct) == 2, "Need exactly two partition values"
        assert partition_pct[0] <= partition_pct[1], "Partition must be monotonic"
    else:
        partition_pct = [0.8, 0.9]

    partition1 = int(partition_pct[0] * len(words))
    partition2 = int(partition_pct[1] * len(words))

    words_training = words[:partition1]
    _rng.shuffle(words_training)
    words_dev = words[partition1:partition2]
    _rng.shuffle(words_dev)
    words_testing = words[partition2:]
    _rng.shuffle(words_testing)

    Xtr, Ytr = build_dataset(words_training, hyperparams)
    Xdev, Ydev = build_dataset(words_dev, hyperparams)
    Xte, Yte = build_dataset(words_testing, hyperparams)

    return DatasetSplit(
        train=(Xtr, Ytr),
        dev=(Xdev, Ydev),
        test=(Xte, Yte),
    )


def get_initial_parameters(
    hyperparams: Hyperparameters, seed: int = None
) -> tuple[Tensor, ...]:
    """Returns (C, W1, b1, W2, b1) as a tuple of tensors."""
    if seed is not None:
        _rng = torch.Generator().manual_seed(seed)
    else:
        _rng = None

    C: Tensor = torch.randn(
        (hyperparams.CHARSET_SIZE, hyperparams.EMBEDDING_SIZE),
        requires_grad=True,
        generator=_rng,
    )

    W1: Tensor = torch.randn(
        (
            hyperparams.EMBEDDING_SIZE * hyperparams.CONTEXT_LENGTH,
            hyperparams.NUM_HIDDEN_NEURONS,
        ),
        requires_grad=True,
        generator=_rng,
    )

    b1: Tensor = torch.randn(
        hyperparams.NUM_HIDDEN_NEURONS,
        requires_grad=True,
        generator=_rng,
    )

    W2: Tensor = torch.randn(
        (hyperparams.NUM_HIDDEN_NEURONS, hyperparams.CHARSET_SIZE),
        generator=_rng,
        requires_grad=True,
    )

    b2: Tensor = torch.randn(
        hyperparams.CHARSET_SIZE, requires_grad=True, generator=_rng
    )

    return (C, W1, b1, W2, b2)


def training_step(
    X: Tensor,
    Y: Tensor,
    parameters: tuple[Tensor, ...],
    hyperparams: Hyperparameters,
    learning_rate=None,
) -> Tensor:
    """
    Takes in a batch of X and Y and returns the loss.

    If learning rate is None then the LR from the hyperparameters is used.

    Modifies the passed parameters.

    Returns the loss.
    """
    C, W1, b1, W2, b2 = parameters
    embedding: Tensor = C[X]
    embedding_view: Tensor = embedding.view(
        -1, hyperparams.EMBEDDING_SIZE * hyperparams.CONTEXT_LENGTH
    )
    h: Tensor = (embedding_view @ W1 + b1).tanh()
    logits: Tensor = h @ W2 + b2

    loss: Tensor = F.cross_entropy(logits, Y)

    for param in parameters:
        param.grad = None

    loss.backward()

    learning_rate: float = learning_rate or hyperparams.LEARNING_RATE
    for param in parameters:
        param.data -= learning_rate * param.grad

    for param in parameters:
        param.grad = None

    return loss


def train(
    X: Tensor,
    Y: Tensor,
    parameters: tuple[Tensor, ...],
    hyperparams: Hyperparameters,
    track_loss: bool = False,
) -> Optional[list[float]]:
    losses: list[float] = []
    for iteration in range(hyperparams.NUM_EPOCHS):
        # Minibatch
        idxs = torch.randint(0, X.shape[0], (hyperparams.BATCH_SIZE,))

        loss = training_step(X[idxs], Y[idxs], parameters, hyperparams)
        print(
            f"({str(iteration).rjust(len(str(hyperparams.NUM_EPOCHS)))}/{hyperparams.NUM_EPOCHS}) - Loss: {loss:.3f}"
        )
        if track_loss:
            losses.append(loss.item())

    if track_loss:
        return losses


def sample_token(params: tuple[Tensor, ...], hyperparams: Hyperparameters) -> list[int]:
    """
    Generates one word as a list of integers corresponding to the character ids.
    """
    out: list[int] = []
    context = [0] * hyperparams.CONTEXT_LENGTH
    while True:
        emb: Tensor = params.C[[context]]
        emb_view: Tensor = emb.view(
            -1, hyperparams.EMBEDDING_SIZE * hyperparams.CONTEXT_LENGTH
        )
        h: Tensor = (emb_view @ params.W1 + params.b1).tanh()
        logits: Tensor = h @ params.W2 + params.b2
        probs: Tensor = F.softmax(logits, dim=1)

        idx: int = torch.multinomial(probs, num_samples=1).item()
        context: list[int] = context[1:] + [idx]

        out.append(idx)
        if idx == 0:
            break


def sample_word(params: tuple[Tensor, ...], hyperparams: Hyperparameters) -> str:
    """
    Generates one word as a string.
    """
    itos, _ = get_itos_stoi()
    return "".join(itos[i] for i in sample_token(params, hyperparams))


def sample_word_list(
    params: tuple[Tensor, ...], hyperparams: Hyperparameters, num: int = 10
) -> list[str]:
    """
    Generates a list of words as strings.
    """
    itos, _ = get_itos_stoi()
    return [
        "".join(itos[i] for i in sample_token(params, hyperparams)) for _ in range(num)
    ]
