import logging


from ignite.engine import Events, create_supervised_evaluator
import ignite.distributed as idist
import ignite
import wandb
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