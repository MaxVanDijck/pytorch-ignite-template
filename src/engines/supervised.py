import logging
import os

import hydra
import ignite
import ignite.distributed as idist
import torch
import torch.nn as nn
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
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
    optimizer = idist.auto_optim(hydra.utils.instantiate(config.engine.optimizer, params=model.parameters(), lr=config.params.learning_rate))
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

    # define trainer engine
    engine = Engine(train_step)

    # VALIDATION
    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(idist.device()), batch[1].to(idist.device())
            y_pred = model(x)
            return y_pred, y

    # define evaluation engine
    evaluator = Engine(validation_step)

    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }
    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    # progress bar
    if idist.get_rank() == 0:
        pbar = ProgressBar()
        pbar.attach(engine, output_transform=lambda x: {"batch loss": x})
        pbar.attach(evaluator, metric_names="all")
        if config.engine.log_to_wandb:
            wandb_logger = WandBLogger()
            wandb_logger.attach_output_handler(
                engine, 
                event_name=Events.ITERATION_COMPLETED, 
                tag="train", 
                output_transform=lambda x: {"batch loss": x}, 
                global_step_transform=lambda *_: engine.state.iteration
            )
            wandb_logger.attach_output_handler(
                evaluator, 
                event_name=Events.EPOCH_COMPLETED, 
                tag="validation", 
                metric_names="all", 
                global_step_transform=lambda *_: engine.state.iteration
            )
            engine.add_event_handler(Events.COMPLETED, wandb.finish)


    # add anything we may want to access in callbacks to engine state
    engine.state.optimizer = optimizer
    engine.state.criterion = criterion
    engine.state.evaluator = evaluator

    # run evaluation on epoch completed
    def run_evaluation(engine):
        engine.state.evaluator.run(engine.state.dataloaders.val)

    engine.add_event_handler(Events.EPOCH_COMPLETED, run_evaluation)

    # model checkpointing
    to_save = {
        'model': model,
        'engine': engine
    }
    handler = ignite.handlers.checkpoint.Checkpoint(
        to_save, 
        os.getcwd(),
        n_saved=3,
        filename_prefix='best',
        score_function=lambda engine: engine.state.metrics["accuracy"],
        score_name="accuracy",
    )
    engine.state.evaluator.add_event_handler(Events.COMPLETED, handler)
    log.info(f"Saving best models to: {os.getcwd()}")

    return engine

