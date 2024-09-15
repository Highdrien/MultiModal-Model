import os
import sys
from os.path import dirname as up
from typing import Iterator, Tuple

import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from model.basemodel import Model, BaseModel


class MultimodalClassifier(Model):
    def __init__(
        self,
        basemodel: dict[str, BaseModel],
        last_hidden_size: int,
        freeze_basemodel: bool = True,
        num_classes: bool = 2,
    ) -> None:
        """Multimodal Model Classifier
        pass data like {'text': tensor, ...} in basemodel, then concatenate them
        and pass in 2 dense layers

        Args:
            basemodel (dict[str, BaseModel]):
                exemple: {'text': BertClassifier, 'audio': Wav2Vec2Classifier,
                    'video': LSTMClassifier}
                if the model take text, audi and video

            last_hidden_size (int):
                output size of the second last layers (and the input size of the last one)

            freeze_basemodel (bool):
                freeze or not the parameter of baseline model

            num_classes (int):
                simply the number of classes
        """
        super().__init__()

        self.keys = list(basemodel.keys())

        if not all(element in ["text", "video", "audio"] for element in self.keys):
            raise ValueError(
                "all keys of basemodel must be text, audio or video. But the keys are ",
                f"{list(basemodel.keys())}",
            )

        self.basemodel = basemodel
        first_hidden_size = 0
        print(f"Multimodal model which take {self.keys}")

        for model in self.basemodel.values():
            model.put_last_layer(last_layer=False)
            first_hidden_size += model.get_hidden_size()

            if freeze_basemodel:
                for param in model.parameters():
                    param.requires_grad = False

        self.fc1 = nn.Linear(
            in_features=first_hidden_size, out_features=last_hidden_size
        )
        self.fc2 = nn.Linear(in_features=last_hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, data: dict[str, Tensor]) -> Tensor:
        """
        Forward pass through the multimodal model.
        Args:
            data (dict[str, Tensor]): A dictionary containing the input data with the
                following keys:
                - `text`: Tensor of shape (B, sequence_size) and dtype torch.int64
                - `audio`: Tensor of shape (B, audio_length) and dtype torch.float32
                - `frames`: Tensor of shape (B, num_frames, num_features) and dtype torch.float32
        Returns:
            Tensor: The output logits of shape (B, num_classes) and dtype torch.float32.
        """
        baseline_output = []
        for key in self.keys:
            baseline_output.append(self.basemodel[key].forward(data[key]))

        x = torch.cat(tuple(baseline_output), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """
        Returns an iterator over module parameters, yielding both the name of the parameter
        as well as the parameter itself.

        Args:
            prefix (str, optional): A prefix to prepend to all parameter names. Defaults to "".
            recurse (bool, optional): If True, then yields parameters of this module and all
                                      submodules. Defaults to True.
            remove_duplicate (bool, optional): If True, removes duplicate parameters.
                Defaults to True.

        Yields:
            Iterator[Tuple[str, Parameter]]:
                An iterator over tuples containing the name of the parameter and
                the parameter itself.
        """
        if recurse:
            for model in self.basemodel.values():
                yield from model.named_parameters(prefix, recurse, remove_duplicate)

        yield from super().named_parameters(prefix, True, remove_duplicate)
        # yield from self.fc1.named_parameters(prefix, recurse, remove_duplicate)
        # yield from self.fc2.named_parameters(prefix, recurse, remove_duplicate)

    def to(self, device: torch.device):
        """
        Moves the model and its sub-models to the specified device.

        Args:
            device (torch.device): The device to move the model to (e.g., 'cpu' or 'cuda').

        Returns:
            self: The instance of the model after being moved to the specified device.
        """
        super().to(device)
        for model in self.basemodel.values():
            model = model.to(device)
        return self

    def eval(self) -> None:
        """
        Evaluates each base model in the multimodal model.

        This method sets each base model to evaluation mode, which typically affects
        layers like dropout and batch normalization, ensuring that they behave
        appropriately during inference.

        Returns:
            self: The instance of the multimodal model with all base models set to evaluation mode.
        """
        for model in self.basemodel.values():
            model = model.eval()
        return self

    def train(self) -> None:
        """
        Trains each base model in the `basemodel` dictionary.

        This method iterates over all models stored in the `basemodel` dictionary,
        calls their `train` method, and updates the model in the dictionary.

        Returns:
            self: The instance of the class after training the models.
        """
        for model in self.basemodel.values():
            model = model.train()
        return self


if __name__ == "__main__":
    import yaml
    from icecream import ic
    from easydict import EasyDict
    from get_model import get_model

    stream = open("config/config.yaml", "r")
    config = EasyDict(yaml.safe_load(stream))

    BATCH_SIZE = config.learning.batch_size
    SEQUENCE_SIZE = config.data.sequence_size
    VIDEO_SIZE = config.data.num_frames
    AUDIO_SIZE = config.data.audio_length
    NUM_FEATURES = config.data.num_features

    config.task = "multi"
    model = get_model(config)
    ic(model)

    # text = torch.randint(0, 100, (BATCH_SIZE, SEQUENCE_SIZE))
    # audio = torch.rand((BATCH_SIZE, AUDIO_SIZE, 2))
    # frames = torch.rand((BATCH_SIZE, VIDEO_SIZE, NUM_FEATURES, 2))

    # ic(text.shape, text.dtype)
    # ic(audio.shape, audio.dtype)
    # ic(frames.shape, frames.dtype)

    # data = {'text': text,
    #         'video': frames,
    #         'audio': audio}

    # y = model.forward(data)
    # ic(y.shape, y.dtype)

    # print('\nname and param with recure True')
    # for name, param in model.named_parameters(recurse=True):
    #     print(name, param.shape)

    print("\nname and param with recure False")
    for name, param in model.named_parameters(recurse=False):
        print(name, param.shape)

    print("\nget learned parameter")
    for name, param in model.get_only_learned_parameters().items():
        print(name, param.shape)
