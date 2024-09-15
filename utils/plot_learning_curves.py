import os
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def print_loss_and_metrics(
    train_loss: float,
    val_loss: float,
    metrics_name: List[str],
    train_metrics: List[float],
    val_metrics: List[float],
) -> None:
    """
    Print loss and metrics for training and validation.

    Args:
        train_loss (float): The loss value for the training set.
        val_loss (float): The loss value for the validation set.
        metrics_name (List[str]): A list of metric names.
        train_metrics (List[float]): A list of metric values for the training set.
        val_metrics (List[float]): A list of metric values for the validation set.
    """
    print(f"{train_loss = }")
    print(f"{val_loss = }")
    for i in range(len(metrics_name)):
        print(
            f"{metrics_name[i]} -> train: {train_metrics[i]:.3f}   val:{val_metrics[i]:.3f}"
        )


def save_learning_curves(path: str) -> None:
    """
    Generates and saves learning curve plots for training and validation metrics.
    The function reads the results from the specified file path, extracts the epochs,
    training metrics, and validation metrics, and then plots these metrics against
    the epochs. Each plot is saved as a PNG file in the same directory as the results file.

    Args:
        path (str): The file path to the results file containing the metrics.
    """
    result, names = get_result(path)

    epochs = result[:, 0]
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel("epoch")
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + ".png"))
        plt.close()


def get_result(path: str) -> Tuple[List[float], List[str]]:
    """
    Reads a CSV file containing training logs and returns the results and column names.

    Args:
        path (str): The directory path where the 'train_log.csv' file is located.

    Returns:
        Tuple[List[float], List[str]]: A tuple containing:
            - A list of lists with the training log values converted to floats.
            - A list of column names from the CSV file.
    """
    with open(os.path.join(path, "train_log.csv"), "r") as f:
        names = f.readline()[:-1].split(",")
        result = []
        for line in f:
            result.append(line[:-1].split(","))

        result = np.array(result, dtype=float)
    f.close()
    return result, names


if __name__ == "__main__":
    logs_path = "logs"
    save_learning_curves(path=os.path.join(logs_path, "audio_1"))
