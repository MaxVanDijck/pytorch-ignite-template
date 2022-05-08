import logging
import logging

import os
import logging
import hydra
from omegaconf import DictConfig
from ignite.utils import manual_seed
from ignite.distributed.auto import auto_dataloader

log = logging.getLogger(__name__)

def train(config: DictConfig):
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

    # Create Train and Validation Datasets
    train_dataset = hydra.utils.instantiate(config.train)
    log.info(f"Train dataset: {train_dataset}")
    val_dataset = hydra.utils.instantiate(config.val)
    log.info(f"Validation dataset: {val_dataset}")

    # Create Train and Validation DataLoaders
    train_loader = auto_dataloader(train_dataset, batch_size=config.params.batch_size)
    log.info(f"Train dataloader: {train_loader}")
    val_loader = auto_dataloader(val_dataset, batch_size=config.params.batch_size)
    log.info(f"Validation dataloader: {val_loader}")

    # Create Model
    model = hydra.utils.instantiate(config.model)
    log.info(f"Model: {model}")

    