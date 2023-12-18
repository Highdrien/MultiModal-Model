import os
import torch
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time
import numpy as np
from easydict import EasyDict
from icecream import ic
import sys
from typing import Tuple

sys.path.append(os.path.join(sys.path[0], '..'))

from src.dataloader.dataloader import create_dataloader
from src.model.get_model import get_model
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

    save_experiment = config.save_experiment
    ic(save_experiment)
    if save_experiment:
        logging_path = train_logger(config)
        best_val_loss = 10e6

    # metrics = Metrics(config=config.metrics, device=device)
    # num_metrics = metrics.num_metrics

    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        ic(epoch)
        train_loss = 0
        train_range = tqdm(train_generator)
        # train_metrics = np.zeros(num_metrics)

        # Training
        for text, audio, video, label in train_range:

            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            label = label.to(device)

            ic(audio.shape)
            
            y = forward(model=model, x=(text, audio, video), task=config.task)
                
            loss = criterion(y, label)

            train_loss += loss.item()
            # train_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {loss.item():.4f}")
            train_range.refresh()


        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_range = tqdm(val_generator)
        # val_metrics = np.zeros(num_metrics)

        with torch.no_grad():
            
            for text, audio, video, label in val_range:
                
                text = text.to(device)
                audio = audio.to(device)
                video = video.to(device)
                label = label.to(device)

                y = forward(model=model, x=(text, audio, video), task=config.task, device=device)
                    
                loss = criterion(y, label)

                # y_pred = torch.nn.functional.softmax(y_pred, dim=1)
                
                val_loss += loss.item()
                # val_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

                val_range.set_description(f"VAL  -> epoch: {epoch} || loss: {loss.item():.4f}")
                val_range.refresh()
        
        scheduler.step()

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        # train_metrics = train_metrics / n_train
        # val_metrics = val_metrics / n_val
        
        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss)
            
            if config.learning.save_checkpoint and val_loss < best_val_loss:
                print('save model weights')
                torch.save(model.state_dict(), os.path.join(logging_path, 'checkpoint.pt'))
                best_val_loss = val_loss
        
        ic(best_val_loss)     

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")

    # if save_experiment and config.learning.save_learning_curves:
    #     save_learning_curves(logging_path)


def forward(model: torch.nn.Module,
            x: Tuple[Tensor, Tensor, Tensor],
            task: str
            ) -> Tensor:
    text, audio, video = x

    if task == 'text':
        y = model.forward(text)
    
    if task == 'audio':
        y = model.forward(audio)
    
    if task == 'video':
        y = model.forward(video)
    
    if task == 'all':
        y = model.forward(text=text, audio=audio, frames=video)
    
    return y
    









if __name__ == '__main__':
    import yaml
    try:
        stream = open(file=os.path.join('..', 'config', 'config.yaml'), mode='r')
        config = EasyDict(yaml.safe_load(stream))
        config.data.path = os.path.join('..', config.data.path)
    except:
        stream = open(file=os.path.join('config', 'config.yaml'), mode='r')
        config = EasyDict(yaml.safe_load(stream))

    ic(config)
    train(config=config)