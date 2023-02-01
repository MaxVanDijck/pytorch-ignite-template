import logging
import os

import hydra
import ignite
import ignite.distributed as idist
import torch.nn as nn
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from omegaconf import DictConfig
from torch.cuda.amp import autocast

log = logging.getLogger(__name__)

def create_engine(model: nn.Module, config: DictConfig):
    """
    Combines the model and config into an engine containing a train and validation loop

    Any extra objects should be added to the engine state such as:
    - Optimizers
    - Loss functions
    This is to ensure we can access these objects 
    through callbacks in train_pipeline.py
    """
    # TRAIN
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

    # VALIDATION
    evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": ignite.metrics.Accuracy(), "loss": ignite.metrics.Loss(engine.state.criterion)},
        device=idist.device(),
    )
    engine.state.evaluator = evaluator
    def function(engine, log_to_wandb=config.engine.log_to_wandb):
        state = engine.state.evaluator.run(engine.state.dataloaders.val)
        if idist.get_rank() == 0:
            log.info(state.metrics)
            if log_to_wandb:
                wandb.log(state.metrics)
                log.info("Logged to Wandb")
    engine.add_event_handler(Events.EPOCH_COMPLETED, function)

    # save best 3 models based upon evaluation
    to_save = {
        'model': model,
        'engine': engine
    }
    handler = ignite.handlers.checkpoint.Checkpoint(
        to_save, 
        os.getcwd(),
        n_saved=3,
        filename_prefix='best',
        score_name="accuracy",
    )
    log.info(f"Saving best models to: {os.getcwd()}")
    engine.state.evaluator.add_event_handler(Events.COMPLETED, handler)

    return engine

