import os
import sys
import time
import numpy as np
from tqdm import tqdm
from icecream import ic
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.metrics import Metrics
from src.model.basemodel import Model
from src.model.get_model import get_model
from src.dataloader.dataloader import create_dataloader
from utils.plot_learning_curves import save_learning_curves
from config.config import train_logger, train_step_logger


def train(config: EasyDict) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available() and config.learning.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ic(device)

    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    val_generator = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator)
    ic(n_train, n_val)

    # Get model
    model = get_model(config)
    model = model.to(device)
    ic(model)
    ic(model.get_number_parameters())
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milesstone, gamma=config.learning.gamma)

    # Get Metrics
    metrics = Metrics(config=config)
    metrics.to(device)

    save_experiment = config.save_experiment
    ic(save_experiment)
    if save_experiment:
        logging_path = train_logger(config)
        best_val_loss = 10e6


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        ic(epoch)
        train_loss = 0
        train_range = tqdm(train_generator)
        train_metrics = np.zeros(metrics.num_metrics)

        # Training
        for i, (data, y_true) in enumerate(train_range):

            dict_to_device(data, device)
            y_true = y_true.to(device)
            y_pred = forward(model=model, data=data, task=config.task)
                
            loss = criterion(y_pred, y_true)

            train_loss += loss.item()
            train_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()
        

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_range = tqdm(val_generator)
        val_metrics = np.zeros(metrics.num_metrics)

        with torch.no_grad():
            
            for i, (data, y_true) in enumerate(val_range):
                
                dict_to_device(data, device)
                y_true = y_true.to(device)
                y_pred = forward(model=model, data=data, task=config.task)
                    
                loss = criterion(y_pred, y_true)                
                val_loss += loss.item()

                val_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
                val_range.refresh()
        
        scheduler.step()

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_val
        
        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics)
            
            if val_loss < best_val_loss:
                print('save model weights')
                torch.save(model.get_only_learned_parameters(),
                           os.path.join(logging_path, 'checkpoint.pt'))
                best_val_loss = val_loss
        
            ic(best_val_loss)     

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")

    if save_experiment and config.learning.save_learning_curves:
        save_learning_curves(logging_path)


def forward(model: Model, data: dict[str, Tensor], task: str) -> Tensor:
    """ forward data in the model accroding the task """
    if task != 'all':
        return model.forward(data[task])
    else:
        return model.forward(text=data['text'], audio=data['audio'], frames=data['video'])

    

def dict_to_device(data: dict[str, Tensor], device: torch.Tensor) -> None:
    """ load all the data.values in the device """
    for key in data.keys():
        data[key] = data[key].to(device)
    







if __name__ == '__main__':
    import yaml
    stream = open(file=os.path.join('config', 'config.yaml'), mode='r')
    config = EasyDict(yaml.safe_load(stream))

    ic(config)
    train(config=config)