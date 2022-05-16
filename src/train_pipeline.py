import os
import logging
import ignite
import torch
import hydra
from omegaconf import DictConfig
from ignite.utils import manual_seed
import ignite.distributed as idist
from ignite.engine import create_supervised_evaluator, Events

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
    if config.data.train.root and not os.path.isabs(config.data.train.root):
        config.data.train.root = os.path.join(hydra.utils.get_original_cwd(), config.data.train.root)
        log.info(f"Train root set to {config.data.train.root}")

    if config.data.val.root and not os.path.isabs(config.data.val.root):
        config.data.val.root = os.path.join(hydra.utils.get_original_cwd(), config.data.val.root)
        log.info(f"Val root set to {config.data.val.root}")

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225)),])

    train_dataset = hydra.utils.instantiate(config.data.train, transform=transform)
    log.info(f"Train dataset: {train_dataset}")
    val_dataset = hydra.utils.instantiate(config.data.val, transform=transform)
    log.info(f"Validation dataset: {val_dataset}")

    # Create Train and Validation DataLoaders
    train_loader = idist.auto_dataloader(train_dataset, batch_size=config.data.batch_size)
    log.info(f"Train dataloader: {train_loader}")
    val_loader = idist.auto_dataloader(val_dataset, batch_size=config.data.batch_size)
    log.info(f"Validation dataloader: {val_loader}")

    # Create Model
    model = idist.auto_model(hydra.utils.instantiate(config.model))
    log.info(f"Model: {model}")

    # Create Optimizer
    optimizer = idist.auto_optim(hydra.utils.instantiate(config.optimizer, params = model.parameters()))
    log.info(f"Optimizer: {optimizer}")

    # Create Loss Function
    loss_fn = hydra.utils.instantiate(config.loss).to(idist.device())
    log.info(f"Loss Function: {loss_fn}")

    # TODO: add ignite handlers

    # TODO: add loggers

    # Create Trainer
    trainer = hydra.utils.instantiate(
        config.trainer, model, optimizer, loss_fn)
    log.info(f"Trainer: {trainer}")

    # Create Evaluator
    evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": ignite.metrics.Accuracy(), "loss": ignite.metrics.Loss(loss_fn)},
        device=idist.device(),
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def evaluate_model():
        state = evaluator.run(val_loader)
        if idist.get_rank() == 0:
            print(state.metrics)

    # Run Trainer
    trainer.run(train_loader, max_epochs=config.data.epochs)
    log.info("Training finished")

    