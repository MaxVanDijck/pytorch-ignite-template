import logging
import os

import hydra
import ignite.distributed as idist
import torch
from ignite.utils import manual_seed
from omegaconf import DictConfig

from src.components import Dataloaders

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

    # Create Model and Load model if applicable
    model = idist.auto_model(hydra.utils.instantiate(config.model))
    log.info(f"Model: {model}")

    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint), strict=False)


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

    # Create Engine
    engine = hydra.utils.instantiate(config.engine.engine, model, config)

    # add anything we may want to access in callbacks to engine state
    engine.state.dataloaders = dataloaders
    engine.state.model = model

    # Add callbacks to engine
    if isinstance(config.callbacks, DictConfig):
        for callback in config.callbacks.values():
            log.info(f"Initializing Callback: {callback}")
            hydra.utils.instantiate(callback, engine)

    # Run Trainer
    engine.run(dataloaders.train, max_epochs=config.params.epochs)
    log.info("Training finished")

    