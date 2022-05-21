import os
import logging
from dataclasses import dataclass
from src.components import Dataloaders

import hydra
from omegaconf import DictConfig
import ignite
import ignite.distributed as idist
from ignite.utils import manual_seed
from ignite.engine import create_supervised_evaluator, Events

log = logging.getLogger(__name__)

def train(local_rank, config: DictConfig):
    # set seed for torch, random and numpy
    if config.seed:
        manual_seed(config.seed)
        log.info(f"Random Seed set to {config.seed}")

    # Convert checkpoint to absolute path if necessary
    if config.checkpoint and not os.path.isabs(config.checkpoint):
        config.checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), config.checkpoint
        )
        log.info(f"Checkpoint set to {config.checkpoint}")

    # Create Model
    model = idist.auto_model(hydra.utils.instantiate(config.model))
    log.info(f"Model: {model}")


    # Create Datasets
    if config.data.datasets.root and not os.path.isabs(config.data.datasets.root):
        config.data.datasets.root = os.path.join(hydra.utils.get_original_cwd(), config.data.datasets.root)

    datasets = hydra.utils.instantiate(config.data.datasets)


    # Create Dataloaders
    dataloaders = Dataloaders(
        train = idist.auto_dataloader(datasets.train, batch_size = config.params.batch_size) if datasets.train else None,
        val = idist.auto_dataloader(datasets.val, batch_size = config.params.batch_size) if datasets.val else None,
        test = idist.auto_dataloader(datasets.test, batch_size = config.params.batch_size) if datasets.test else None,
    )

    # Create Optimizer and Loss Function
    optimizer = idist.auto_optim(hydra.utils.instantiate(config.optimizer, params=model.parameters()))
    criterion = hydra.utils.instantiate(config.criterion).to(idist.device())

    # Create Engine
    engine = hydra.utils.instantiate(config.engine, model, optimizer, criterion, config)

    # Create Evaluator
    evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": ignite.metrics.Accuracy(), "loss": ignite.metrics.Loss(hydra.utils.instantiate(config.criterion))},
        device=idist.device(),
    )

    # add anything we may want to access in callbacks to engine state
    engine.state.criterion = criterion
    engine.state.optimizer = optimizer
    engine.state.evaluator = evaluator
    engine.state.dataloaders = dataloaders

    # Add callbacks to engine
    for callback in config.callbacks.values():
        log.info(f"Initializing Callback: {callback}")
        hydra.utils.instantiate(callback, engine)

    # Run Trainer
    engine.run(dataloaders.train, max_epochs=config.params.epochs)
    log.info("Training finished")

    