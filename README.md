# Projet SAM

Our project focuses on multimodal approaches for predicting turn-taking changes in natural conversations. The goals of this project enable us to introduce various concepts of textual, visual, and auditory modality, as well as to compare and explore different multimodal processing models and their fusion.

# Requirements
To run the code you need python (We use python 3.9.13) and packages that is indicate in [`requirements.txt`](requirements.txt).
You can run the following code to install all packages in the correct versions:
```sh
pip install -r requirements.txt
```

# Launch the code

The [`main.py`](main.py) script is the main entry point for this project. It accepts several command-line arguments to control its behavior:

- `--mode` or `-m`: This option allows you to choose a mode between 'train', and 'test'.
- `--config_path` or `-c`: This option allows you to specify the path to the configuration file for training. The default is [`config/config.yaml`](config/config.yaml).
- `--path` or `-p`: This option allows you to specify the experiment path for testing, prediction, or generation.
- `--task` or `-t`: This option allows you to specify the task for the model. This will overwrite the task specified in the configuration file for training. It's can be `text`, `audio`, `video`, or `multi`, that will train a new experiments with this type of data.

## Mode
Here's what each mode does:

- [`train`](src/train.py): Trains a model using the configuration specified in the `--config_path` and the task specified in `--task`.
- [`test`](src/test.py): Tests the model specified in the `--path`. You must specify a path.

## Configuration

The configuration of the model and the training process is done through a YAML file. You can specify the path to this file with the `--config_path` option. The default path is [`config/config.yaml`](config/config.yaml).

The configuration file includes various parameters such as the learning rate, batch size, number of epochs, etc.

## Help

To get a list of all available options, you can use the `-h` or `--help` option:

```sh
python main.py --help
```

This will display a help message with a description of all available options.

## Example
Here's an example of how to use the script to train a model:

```sh
python main.py --mode train --config_path config/config.yaml --task text
```

This command will train a model using the configuration specified in [`config/config.yaml`](config/config.yaml) with a `task=text`.

Here's an example of how to run a test on the experiment separete:

```sh
python main.py --mode test --path logs/multi_4
```

# Models
## Unimodal Models
### TEXT
<p align="center"><img src=report\image_model\model_text.png><p>

### AUDIO
<p align="center"><img src=report\image_model\model_audio.png><p>

### VIDEO
<p align="center"><img src=report\image_model\model_video.png><p>

## Multimodal Models
### LATE FUSION
<p align="center"><img src=report\image_model\late_fusion.png><p>

### EARLY FUSION
<p align="center"><img src=report\image_model\early_fusion.png><p>

# Results

| Modèle            | Accuracy | Precision | Rappel | $f_1$ score |
|-------------------|----------|-----------|--------|-------------|
| *TEXT*            | 82.8     | 41.3      | 50.0   | 45.3        |
| *AUDIO*           | 47.1     | 48.5      | 47.4   | 41.5        |
| *VIDEO*           | **82.9** | 41.4      | 50.0   | 45.2        |
| *LATE FUSION*     | 78.5     | **50.6**  | 50.1   | **48.8**    |
| *EARLY FUSION*    | **82.9** | 43.6      | **50.2** | 45.7      |

*Table 1: Résultats de test des modèles. Les *LATE* et *EARLY FUSION* n'utilisent pas le modèle *VIDEO*. (Référence : [tab: test])*
