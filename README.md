# Projet SAM

Informations:
- data: attendre les infos
- deadline: ~fin janvier
- rapport: parler de notre procédure, nos diffucultées. Ne pas faire de de state of the arts
- Utiliser la voix, les images, le texte (script)

Bibliographie:

- Solutions de Voice Turn Taking: https://huggingface.co/blog/fine-tune-wav2vec2-english finetuning de wave2vec
- Papier sur le Wave2Vec finetuning: https://arxiv.org/abs/2109.15053
- Papier sur la détection de turn taking: https://arxiv.org/pdf/2208.13321.pdf

- Text turn taking: https://github.com/speechbrain/speechbrain
- https://www.analyticsvidhya.com/blog/2022/06/automatic-speech-recognition-using-wav2vec2/

# README

This project is a multimodal processing system. The main entry point of the project is [`main.py`](command:_github.copilot.openRelativePath?%5B%22main.py%22%5D "main.py").

## Features

- Training mode: Train the model with the provided dataset.
- Data mode: Process and prepare the data for training.

## Usage

You can run the [`main.py`](command:_github.copilot.openRelativePath?%5B%22main.py%22%5D "main.py") script from the command line with various options.

### Training Mode

To train the model, use the [`--mode`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fdataloader%2Fdataloader.py%22%2C%22--mode%22%5D "src/dataloader/dataloader.py") option with the value `train`. You can specify the path to the configuration file with the `--config_path` option. If not provided, it defaults to [`config/config.yaml`](command:_github.copilot.openRelativePath?%5B%22config%2Fconfig.yaml%22%5D "config/config.yaml").

Example for training:

```sh
python main.py --mode train --config_path config/my_config.yaml
```

Example for testing:
    
```sh
python main.py --mode test --path config/my_config.yaml
```
Replace `config/my_config.yaml` with the path to your configuration file. For exemple `logs/multi_0`.



### Data Mode

To process and prepare the data, use the [`--mode`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fdataloader%2Fdataloader.py%22%2C%22--mode%22%5D "src/dataloader/dataloader.py") option with the value [`data`](command:_github.copilot.openRelativePath?%5B%22data%22%5D "data"). You can specify the path to the data with the [`--path`](command:_github.copilot.openSymbolInFile?%5B%22config%2Fconfig.py%22%2C%22--path%22%5D "config/config.py") option.

Example:

```sh
python main.py --mode data --path data/my_data.csv
```

### Task Option

You can specify a task for the model with the `--task` option. This will overwrite the task specified in the config file.

Example:

```sh
python main.py --mode train --task my_task
```

## Configuration

The configuration of the model and the training process is done through a YAML file. You can specify the path to this file with the `--config_path` option. The default path is [`config/config.yaml`](config/config.yaml).

The configuration file includes various parameters such as the learning rate, batch size, number of epochs, etc.

## Help

To get a list of all available options, you can use the `-h` or `--help` option:

```sh
python main.py --help
```

This will display a help message with a description of all available options.