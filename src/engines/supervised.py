import hydra
from omegaconf import DictConfig

import torch.nn as nn
from torch.cuda.amp import autocast

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar


def create_engine(model: nn.Module, config: DictConfig):
    """
    Combines the model and config into an engine

    Any extra objects should be passed to this function such as:
    - Optimizers
    - Loss functions
    This is to ensure we can access these objects 
    through callbacks in train_pipeline.py

    WARNING: If we initilize anything other than the engine here, 
    we will not be able to access it through callbacks
    """

    optimizer = idist.auto_optim(hydra.utils.instantiate(config.engine.optimizer, params=model.parameters()))
    criterion = hydra.utils.instantiate(config.engine.criterion).to(idist.device())
    
    # Define any training logic for iteration update
    def train_step(engine, batch):
        model.train()

        x, y = batch[0].to(idist.device()), batch[1].to(idist.device())

        if config.engine.mixed_precision:
            with autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Define trainer engine
    engine = Engine(train_step)

    # add anything we may want to access in callbacks to engine state
    engine.state.criterion = criterion
    engine.state.optimizer = optimizer


    if idist.get_rank() == 0:
        # Add progress bar showing batch loss value
        ProgressBar().attach(engine, output_transform=lambda x: {"batch loss": x})

    return engine

