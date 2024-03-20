"""Bigram Neural Network."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .bigram import get_itos_stoi


def loss(W: Tensor, xs: Tensor, ys: Tensor):
    """Computes the loss for the bigram neural network."""
    xenc: Tensor = F.one_hot(xs, num_classes=27).float()

    logits: Tensor = xenc @ W
    counts: Tensor = logits.exp()
    probs: Tensor = counts / counts.sum(1, keepdim=True)

    _l2_regularization_weight = 0.01
    _l2_regularization: Tensor = _l2_regularization_weight * (W**2).mean()

    _average_negative_log_loss: Tensor = (
        -probs[torch.arange(xs.nelement()), ys].log().mean()
    )

    return _average_negative_log_loss + _l2_regularization


def training_step(
    W: Tensor, xs: Tensor, ys: Tensor, learning_rate: float = 10, do_print=False
) -> None:
    """Do one gradient descent iteration."""
    _loss = loss(W, xs, ys)
    if do_print:
        print(_loss.item())

    W.grad = None
    _loss.backward()

    W.data -= learning_rate * W.grad


def train_bigram_nn(
    W, xs, ys, max_epochs: int = 20, learning_rate=30, do_print=False
) -> None:
    """Trains the bigram neural network."""
    for _ in range(max_epochs):
        training_step(W, xs, ys, learning_rate=learning_rate, do_print=do_print)


def sample_word(W: Tensor) -> str:
    """Given a weight matrix W, sample a word from the bigram model."""
    itos, _ = get_itos_stoi()
    out: list[str] = []
    idx = 0
    while True:
        xenc: Tensor = F.one_hot(torch.tensor([idx]), num_classes=27).float()
        logits: Tensor = xenc @ W
        counts: Tensor = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        idx: int = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[idx])
        if idx == 0:
            break
    return "".join(out)
