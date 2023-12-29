import os
import sys
import numpy as np
from icecream import ic
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))


class Metrics:
    def __init__(self, config: EasyDict) -> None:
        if config.data.num_classes != 2:
            raise NotImplementedError(f'Attention: only binary accuracy was implemented')
        
        self.metrics = {}
        metrics_name = []
        # test if metrics is in config
        if 'metrics' in config:
            metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
            ic(metrics_name)
        
        # test with metrics there are
        if 'acc' in metrics_name:
            self.metrics['acc'] = Accuracy(task='binary')
        
        if 'precision' in  metrics_name:
            self.metrics['precision'] = Precision(task='binary')
        
        if 'recall' in metrics_name:
            self.metrics['recall'] = Recall(task='binary')
        
        if 'f1' in  metrics_name or 'f1score' in metrics_name:
            self.metrics['f1'] = F1Score(task='binary')
    
        ic(self.metrics)
        self.num_metrics = len(self.metrics)
    
    def compute(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        """ compute all the metrics 
        y_pred and y_true must have shape like (B, 2)
        """
        metrics_value = []
        for metric in self.metrics.values():
            metrics_value.append(metric(y_pred, y_true))
        return np.array(metrics_value)




if __name__ == '__main__':
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))
    B = 4       # batch_size
    seuil = 0.66
    y_true = torch.rand((B, 2))
    y_pred = torch.rand(B, 2)
    y_true[y_true <= seuil] = 0 
    y_true[y_true > seuil] = 1

    ic(y_pred)
    ic(y_true)

    metrics = Metrics(config=config)
    metrics_value = metrics.compute(y_pred=y_pred, y_true=y_true)

    for i, metric_name in enumerate(metrics.metrics.keys()):
        print(f"{metric_name[:6]}\t->\t{metrics_value[i]:.3f}")
        
