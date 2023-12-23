import os
import yaml
import argparse
from icecream import ic
from easydict import EasyDict

from src.train import train


def load_config(path: str='config/config.yaml') -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .yaml was found in", experiment_path)
    
    exit()

IMPLEMENTED = ['train']

def main(options: dict) -> None:

    if options['mode'] not in IMPLEMENTED:
        raise ValueError(f"Expected mode must in {IMPLEMENTED} but found {options['mode']}")

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        if options['task'] is not None:
            config.task.task_name = options['task']
        ic(config)
        train(config)
    
    if options['mode'] == 'baseline':
        config = load_config(options['config_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', '-m', default=None, type=str,
                        help="choose a mode between 'train', 'data'")
    parser.add_argument('--config_path', '-c', default=os.path.join('config', 'config.yaml'),
                        type=str, help="path to config (for training)")
    parser.add_argument('--path', '-p', type=str,
                        help="experiment path (for test, prediction or generate)")
    parser.add_argument('--task', '-t', type=str, default=None,
                        help="task for model (will overwrite the config) for trainning")

    args = parser.parse_args()
    options = vars(args)

    main(options)