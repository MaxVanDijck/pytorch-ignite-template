import wandb
from ignite.engine import Events
from ignite.handlers import LRScheduler
from torch.optim.lr_scheduler import StepLR


def add_lr_scheduler(engine):
    torch_lr_scheduler = StepLR(engine.state.optimizer, step_size=8, gamma=0.1)
    scheduler = LRScheduler(torch_lr_scheduler)
    engine.add_event_handler(Events.EPOCH_COMPLETED, scheduler)


def init_wandb(engine):
    wandb.login()
    wandb.init()