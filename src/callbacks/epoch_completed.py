import logging
import os

import ignite
import ignite.distributed as idist
import wandb
from ignite.engine import Events, create_supervised_evaluator

log = logging.getLogger(__name__)


def print_learning_rate(engine):
    def function(engine):
        log.info(engine.state.optimizer.param_groups[0]["lr"])

    engine.add_event_handler(Events.EPOCH_COMPLETED, function)
    
def evaluate_model(engine, log_to_wandb):
    # Create Evaluator
    evaluator = create_supervised_evaluator(
        engine.state.model,
        metrics={"accuracy": ignite.metrics.Accuracy(), "loss": ignite.metrics.Loss(engine.state.criterion)},
        device=idist.device(),
    )
    engine.state.evaluator = evaluator
    def function(engine, log_to_wandb=log_to_wandb):
        state = engine.state.evaluator.run(engine.state.dataloaders.val)
        if idist.get_rank() == 0:
            log.info(state.metrics)
            if log_to_wandb:
                wandb.log(state.metrics)
                log.info("Logged to Wandb")
    engine.add_event_handler(Events.EPOCH_COMPLETED, function)


    # save best 3 models based upon evaluation
    to_save = {
        'model': engine.state.model,
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