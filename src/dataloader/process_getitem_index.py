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
    all_label: List[bool] = []
    
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
                        text_index: Text_Index,
                        step_frame: Optional[float]=0.040):
    
    data_path = os.path.join(data_path, 'video', 'openface')
    file_list = os.listdir(data_path)
    output = []

    for file in files.keys():
        speakers = files[file]
        dataset = 'cheese'
        filename = f"{dataset}_{file}_{speakers[0]}.csv"

        if filename not in file_list:
            dataset = 'paco'
        
        for speaker in speakers:
            filename = os.path.join(data_path, f"{dataset}_{file}_{speaker}.csv")
            ic(filename)

            df = pd.read_csv(filename)
            frame_id = df['frame'].tolist()
            timestamp = df[' timestamp'].tolist()
            max_timestamp = max(timestamp)
            del df

            for i, (_, _, time) in enumerate(text_index):
                if time < max_timestamp:
                    frame_time = step_frame * (time // step_frame)
                    frame_time = round(frame_time, 3)
                    if frame_time not in timestamp:
                        print(f"attention erreur: {file = }, {time = }, {frame_time = }")
                        exit()
                    index = timestamp.index(frame_time) + 1 # car les images commance à 1
                    
                




if __name__ == '__main__':
    from icecream import ic
    files = {'JAEA': ['JA', 'EA'],
             'JSCL': ['JS', 'CL'],
             'LPMA': ['LP', 'MA'],
             'MDAD': ['MD', 'AD'],
             'RPABN': ['RPA', 'BN']}
    data_path = 'data'
    text_index, all_label = process_text_index(files=files, data_path=data_path)
    # ic(text_index)
    # ic(all_label)
    process_index_video(files=files, data_path=data_path, text_index=text_index)
