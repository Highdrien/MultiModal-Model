import os
import pandas as pd
from typing import List, Tuple, Dict
from transformers import DistilBertTokenizer

import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader

import dataloader.get_data as get_data


class DataGenerator(Dataset):
    def __init__(self,
                 mode: str,
                 data_path: str,
                 load: Dict[str, bool],
                 sequence_size: int,
                 audio_size: int,
                 video_size: int) -> None:
        super().__init__()

        print(f'creating {mode} generator')
        self.mode = mode
        self.data_path = data_path
        self.load = load

        self.sequence_size = sequence_size
        self.audio_size = audio_size
        self.video_size = video_size

        self.df = pd.read_csv(os.path.join(data_path, f"item_{mode}.csv"))
        self.num_data = len(self.df)

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self) -> int:
        return self.num_data
    
    def __getitem__(self, index: int) -> List[Tensor]:
        """ renvoie 4 tensors: text, audio, video et label
        les 3 premiers peuvent valoir None si load[<type>]=False

        text: List[str]     # faire le tokenizer après
        video: Tensor de shape (2 * video_size, 709)
        label: Tensor de shape (2)  # one hot encoding
        """
        line = self.df.loc[index]
        label = torch.tensor(int(line['label']))
        label = one_hot(label, num_classes=2).to(torch.float32)
        text, audio, video = [torch.zeros(1)] * 3

        if self.load['text']:
            text = get_data.get_text(info=line)
            text = self.tokenizer(text)['input_ids'][:self.sequence_size]
            text = torch.tensor(text)

        if self.load['video']:
            s0 = get_data.get_frame(info=line, video_size=self.video_size, speaker=0)
            s1 = get_data.get_frame(info=line, video_size=self.video_size, speaker=1)

            video = torch.concat([s0, s1], dim=0)
        
        return text, audio, video, label
        


def create_dataloader(mode: str, load: Dict[str, bool]) -> DataLoader:
    generator = DataGenerator(mode=mode,
                              data_path='data',
                              load=load, 
                              sequence_size=10,
                              audio_size=1,
                              video_size=10)
    dataloader = DataLoader(generator,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True)
    return dataloader




if __name__ == '__main__':
    LOAD = {'audio': False, 'text': True, 'video': True}
    generator = DataGenerator(mode='val',
                              data_path='data',
                              load=LOAD, 
                              sequence_size=10,
                              audio_size=1,
                              video_size=10)
    print('num data in generator:', len(generator))
    text, audio, video, label = generator.__getitem__(index=32)
    print('text shape:', text.shape)
    print('audio:', audio)
    print('video shape:', video.shape)
    print('label shape:', label.shape)

    # test dataloader
    test_dataloader = create_dataloader(mode='test', load=LOAD)
    text, audio, video, label = next(iter(test_dataloader))
    print('text shape:', text.shape)
    print('audio shape:', audio.shape)
    print('video shape:', video.shape)
    print('label shape:', label.shape)


