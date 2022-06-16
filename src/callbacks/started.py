from ignite.engine import Events

from ignite.handlers import LRScheduler
from torch.optim.lr_scheduler import StepLR

import wandb

def add_lr_scheduler(engine):
    def function(engine):
        torch_lr_scheduler = StepLR(engine.state.optimizer, step_size=8, gamma=0.1)
        scheduler = LRScheduler(torch_lr_scheduler)
        engine.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    engine.add_event_handler(Events.STARTED, function)

def init_wandb(engine):
    def function(engine):
        wandb.login()
        wandb.init()
    engine.add_event_handler(Events.STARTED, function)