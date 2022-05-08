import os
import logging
import torch
import hydra
from omegaconf import DictConfig
from ignite.utils import manual_seed
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim
import ignite.distributed as idist

# TODO: Move Transformations to a separate file
from torchvision.transforms import Compose, Normalize, ToTensor

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
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225)),])

    train_dataset = hydra.utils.instantiate(config.train, transform=transform)
    log.info(f"Train dataset: {train_dataset}")
    val_dataset = hydra.utils.instantiate(config.val, transform=transform)
    log.info(f"Validation dataset: {val_dataset}")

    # Create Train and Validation DataLoaders
    train_loader = idist.auto_dataloader(train_dataset, batch_size=config.params.batch_size)
    log.info(f"Train dataloader: {train_loader}")
    val_loader = idist.auto_dataloader(val_dataset, batch_size=config.params.batch_size)
    log.info(f"Validation dataloader: {val_loader}")

    # Create Model
    model = idist.auto_model(hydra.utils.instantiate(config.model))
    log.info(f"Model: {model}")

    # Create Optimizer
    optimizer = idist.auto_optim(hydra.utils.instantiate(config.optimizer, model.parameters()))
    log.info(f"Optimizer: {optimizer}")

    # Create Loss Function
    loss_fn = hydra.utils.instantiate(config.loss).to(idist.device())

    # TODO: add ignite handlers

    # TODO: add loggers

    # Create Trainer
    trainer = hydra.utils.instantiate(
        config.trainer, model, optimizer, loss_fn, train_loader, val_loader
        )
    log.info(f"Trainer: {trainer}")

    # Run Trainer
    trainer.run(train_loader, max_epochs=config.params.epochs)
    log.info("Training finished")

    