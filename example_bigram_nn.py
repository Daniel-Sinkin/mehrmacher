"""Example Bigram Neural Network."""

import torch
import torch.nn.functional as F
from torch import Tensor

from src.bigram import get_itos_stoi
from src.bigram_nn import sample_word, train_bigram_nn
from src.util import get_names_list


def main():
    itos, stoi = get_itos_stoi()

    xs, ys = [], []
    for w in get_names_list():
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            idx1: str = stoi[ch1]
            idx2: str = stoi[ch2]
            xs.append(idx1)
            ys.append(idx2)

    xs: Tensor = torch.tensor(xs)
    ys: Tensor = torch.tensor(ys)

    xenc: Tensor = F.one_hot(xs, num_classes=27).float()

    W = torch.randn((27, 27), requires_grad=True)
    train_bigram_nn(W, xs, ys, max_epochs=2500, learning_rate=40, do_print=True)

    for iteration in range(20):
        print(f"{iteration:4} : {sample_word(W):25}")


if __name__ == "__main__":
    main()
