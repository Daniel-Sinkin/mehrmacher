import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.util import get_chars, get_itos_stoi, get_names_list


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
