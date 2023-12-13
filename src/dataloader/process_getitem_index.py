import os
import pandas as pd
from typing import List, Dict, Tuple, Optional


Files = Dict[str, List[str]]                # key: file, value: speaker_1, speaker_2
Text_Index = List[Tuple[str, int, float]]   # str: filepath, int: ipu_index, float: time_stop


def process_text_index(files: Files,
                       data_path: str,
                       ignore_first_index: Optional[int] = 0
                       ) -> Tuple[Text_Index, List[bool]]:
    """
    récurere la liste de filepath, ipu_id, time_stop et les labels correspondant
    si ignore_first_index = t, récupère seulement les élements tq ipu_id > t
    """
    text_index: List[Tuple[str, int, float]] = []
    all_label = List[bool] = []
    
    for file in files.keys():
        filepath = os.path.join(data_path, 'transcr', file + '_merge.csv')
        df = pd.read_csv(filepath)
        ipu = df['ipu_id'].tolist()[ignore_first_index:]
        stop = df['stop'].tolist()[ignore_first_index:]
        label = df['turn_after'].tolist()[ignore_first_index:]
        del df
        new_list = [(filepath, ipu[i], stop[i]) for i in range(len(ipu))]
        text_index += new_list
        all_label += label
        del new_list, label
    
    return text_index, all_label


def process_index_video(files: Files,
                        data_path: str,
                        text_index: Text_Index):
    data_path = os.path.join(data_path, 'video', 'openface')
    file_list = os.listdir(data_path)
    output = []

    for file in files.keys():
        speakers = files[file]
        dataset = 'chesse'
        filename = f"{dataset}_{file}_{speakers[0]}.csv"

        if filename not in file_list:
            dataset = 'paco'
        
        for speaker in speakers:
            filename = f"{dataset}_{file}_{speaker}.csv"
            
            df = pd.read_csv(filename)
            # frame_id = df['frame'].tolist()
            timestamp = df['timestamp'].tolist()
            del df

            for _, _, time in text_index:
                #TODO: match les time de text index et les timestamp
                pass





