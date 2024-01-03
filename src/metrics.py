import os
import sys
import numpy as np
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
        
        # test with metrics there are
        if 'acc' in metrics_name:
            self.metrics['acc'] = Accuracy(task='binary')
        
        if 'precision' in  metrics_name:
            self.metrics['precision'] = Precision(task='binary')
        
        if 'recall' in metrics_name:
            self.metrics['recall'] = Recall(task='binary')
        
        if 'f1' in  metrics_name or 'f1score' in metrics_name:
            self.metrics['f1'] = F1Score(task='binary')
    
        self.num_metrics = len(self.metrics)
        self.metrics_name = list(self.metrics.keys())
    
    def compute(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        """ compute all the metrics 
        y_pred and y_true must have shape like (B, 2)
        """
        metrics_value = []
        for metric in self.metrics.values():
            metrics_value.append(metric(y_pred, y_true).item())
        return np.array(metrics_value)

    def __str__(self) -> str:
        return f'Metrics: {self.metrics}'
    
    def to(self, device: torch.device) -> None:
        for key in self.metrics.keys():
            self.metrics[key] = self.metrics[key].to(device)


if __name__ == '__main__':
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))
    B = 4       # batch_size
    seuil = 0.66
    y_true = torch.rand((B, 2))
    y_pred = torch.rand(B, 2)
    y_true[y_true <= seuil] = 0 
    y_true[y_true > seuil] = 1


    metrics = Metrics(config=config)
    print(metrics)
    metrics_value = metrics.compute(y_pred=y_pred, y_true=y_true)
    print(metrics_value)

    for i, metric_name in enumerate(metrics.metrics.keys()):
        print(f"{metric_name[:6]}\t->\t{metrics_value[i]:.3f}")
    
    print(metrics.metrics_name)
        
