import os
import yaml
from typing import Optional
from datetime import datetime
from easydict import EasyDict


def number_folder(path: str, name: str) -> str:
    """
    Generates a new folder name by appending an incremented number to the given base name.
    It checks the existing folders in the specified path and finds the highest numbered folder
    with the given base name, then returns the next number in the sequence.

    Args:
        path (str): The directory path where the folders are located.
        name (str): The base name of the folders to be numbered.

    Returns:
        str: The new folder name with the next number in the sequence.
    """
    elements = os.listdir(path)
    last_index = -1
    for i in range(len(elements)):
        folder_name = name + str(i)
        if folder_name in elements:
            last_index = i
    return name + str(last_index + 1)


def train_logger(
    config: EasyDict, write_train_log: bool = True, copy_config: bool = True
) -> str:
    """
    Creates a logs folder where the configuration is saved in config.yaml and
    a train_log.csv file is created to store loss and metrics values.

    Args:
        config (EasyDict): Configuration dictionary containing training settings.
        write_train_log (bool, optional):
            Flag to determine if train_log.csv should be created. Defaults to True.
        copy_config (bool, optional):
            Flag to determine if config.yaml should be copied. Defaults to True.

    Returns:
        str: Path to the created logs folder.
    """
    path = config.logs if "logs" in config else "logs"
    if not os.path.exists(path):
        os.makedirs(path)
    folder_name = number_folder(path, config.task + "_")
    path = os.path.join(path, folder_name)
    os.mkdir(path)
    print(f"{path = }")

    if write_train_log:
        # create train_log.csv where save the metrics
        with open(os.path.join(path, "train_log.csv"), "w") as f:
            first_line = "step," + config.learning.loss + ",val " + config.learning.loss
            if "metrics" in config.keys():
                for metric in list(filter(lambda x: config.metrics[x], config.metrics)):
                    first_line += "," + metric
                    first_line += ",val " + metric
            f.write(first_line + "\n")
        f.close()

    if copy_config:
        # copy the config
        with open(os.path.join(path, "config.yaml"), "w") as f:
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            f.write("config_metadata: 'Saving time : " + date_time + "'\n")
            for line in config_to_yaml(config):
                f.write(line + "\n")
        f.close()

    return path


def config_to_yaml(config: EasyDict, space: str = "") -> str:
    """
    Converts an EasyDict configuration object to a YAML formatted string.

    Args:
        config (EasyDict): The configuration object to convert.
        space (str, optional): The current indentation level. Defaults to an empty string.

    Returns:
        str: A YAML formatted string representing the configuration.
    """
    intent = " " * 4
    config_str = []
    for key, value in config.items():
        if isinstance(value, EasyDict):
            if len(space) == 0:
                config_str.append("")
                config_str.append(space + "# " + key + " options")
            config_str.append(space + key + ":")
            config_str += config_to_yaml(value, space=space + intent)
        elif isinstance(value, str):
            config_str.append(space + key + ": '" + str(value) + "'")
        elif value is None:
            config_str.append(space + key + ": null")
        elif isinstance(value, bool):
            config_str.append(space + key + ": " + str(value).lower())
        else:
            config_str.append(space + key + ": " + str(value))
    return config_str


def train_step_logger(
    path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: Optional[list[float]] = [],
    val_metrics: Optional[list[float]] = [],
) -> None:
    """
    Logs training and validation metrics to a CSV file.

    Args:
        path (str): The directory path where the log file is stored.
        epoch (int): The current epoch number.
        train_loss (float): The training loss for the current epoch.
        val_loss (float): The validation loss for the current epoch.
        train_metrics (Optional[List[float]]):
            A list of training metrics for the current epoch. Defaults to an empty list.
        val_metrics (Optional[List[float]]):
            A list of validation metrics for the current epoch. Defaults to an empty list.
    """
    with open(os.path.join(path, "train_log.csv"), "a") as file:
        line = str(epoch) + "," + str(train_loss) + "," + str(val_loss)
        for i in range(len(train_metrics)):
            line += "," + str(train_metrics[i])
            line += "," + str(val_metrics[i])
        file.write(line + "\n")
    file.close()


def test_logger(path: str, metrics: list[str], values: list[float]) -> None:
    """
    Creates a log file named 'test_log.txt' in the specified path and writes each
    metric and its corresponding value to the file.

    Args:
        path (str): The directory path where the log file will be created.
        metrics (list[str]): A list of metric names.
        values (list[float]): A list of values corresponding to the metrics.
    """
    with open(os.path.join(path, "test_log.txt"), "a") as f:
        for i in range(len(metrics)):
            f.write(metrics[i] + ": " + str(values[i]) + "\n")


def load_config(path: str = "config/config.yaml") -> EasyDict:
    """
    Load configuration from a YAML file and return it as an EasyDict.

    Args:
        path (str): The path to the YAML configuration file. Defaults to "config/config.yaml".

    Returns:
        EasyDict: The configuration loaded from the YAML file.
    """
    stream = open(path, "r")
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    """
    Finds the configuration file in the given experiment directory.

    Args:
        experiment_path (str): The path to the experiment directory.

    Returns:
        str: The full path to the configuration file.

    Raises:
        FileNotFoundError: If no .yaml file is found in the directory.
        FileNotFoundError: If more than one .yaml file is found in the directory.
    """
    yaml_in_path = list(
        filter(lambda x: x[-5:] == ".yaml", os.listdir(experiment_path))
    )

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        raise FileNotFoundError("ERROR: config.yaml wasn't found in", experiment_path)

    if len(yaml_in_path) > 0:
        raise FileNotFoundError("ERROR: a lot a .yaml was found in", experiment_path)
