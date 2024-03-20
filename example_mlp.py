import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.mlp import Hyperparameters, build_dataset, get_initial_parameters, train
from src.util import get_itos_stoi, get_names_list


def main() -> None:
    hyperparams = Hyperparameters()
    parameters = get_initial_parameters(hyperparams)
    X, Y = build_dataset(words=get_names_list(), hyperparams=hyperparams)

    hyperparams = Hyperparameters()
    hyperparams.BATCH_SIZE = 128
    hyperparams.NUM_EPOCHS = 8_000

    print("Starting to train")
    losses = train(X, Y, parameters, hyperparams, track_loss=True)

    x_plot = list(range(len(losses)))[1000:]
    y_plot = losses[1000:]

    window_size = 100
    moving_average_loss = np.convolve(
        losses, np.ones(window_size) / window_size, mode="valid"
    )
    plot_ma = moving_average_loss[1000 - window_size + 1 :]
    plt.plot(x_plot, y_plot, label="Loss", color="b")
    plt.plot(x_plot, plot_ma, label="Moving average", color="r")

    plt.title("MLP Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig("images/mlp_example_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
