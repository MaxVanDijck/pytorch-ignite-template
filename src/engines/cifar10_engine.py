import hydra
from omegaconf import DictConfig

import torch.nn as nn

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar

def create_engine(model: nn.Module, config: DictConfig):
    """Combines the model and config into an engine"""
    
    optimizer = idist.auto_optim(hydra.utils.instantiate(config.optimizer, params=model.parameters()))
    criterion = hydra.utils.instantiate(config.criterion).to(idist.device())

    # Define any training logic for iteration update
    def train_step(engine, batch):
        model.train()

        x, y = batch[0].to(idist.device()), batch[1].to(idist.device())

        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Define trainer engine
    engine = Engine(train_step)

    if idist.get_rank() == 0:
        # Add progress bar showing batch loss value
        ProgressBar().attach(engine, output_transform=lambda x: {"batch loss": x})

    return engine

