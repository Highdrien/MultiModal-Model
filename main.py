import os
import argparse
from icecream import ic

from config.config import load_config, find_config
from src.train import train
from src.test import test



IMPLEMENTED = ['train', 'test', 'baseline']

def main(options: dict) -> None:

    if options['mode'] not in IMPLEMENTED:
        raise ValueError(f"Expected mode must in {IMPLEMENTED} but found {options['mode']}")

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        if options['task'] is not None:
            config.task = options['task']
        ic(config)
        train(config)
    
    if options['mode'] == 'test':
        if options['path'] is None:
            raise ValueError(f'you must specify the path of the experiment that you want to test')
        config = load_config(find_config(experiment_path=options['path']))
        ic(config)
        test(config, logging_path=options['path'])
    
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