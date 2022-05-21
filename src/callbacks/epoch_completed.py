import logging


from ignite.engine import Events
import ignite.distributed as idist

log = logging.getLogger(__name__)


def print_learning_rate(engine):
    def function(engine):
        log.info(engine.state.optimizer.param_groups[0]["lr"])

    engine.add_event_handler(Events.EPOCH_COMPLETED, function)
    
def evaluate_model(engine):
    def function(engine):
        state = engine.state.evaluator.run(engine.state.dataloaders.val)
        if idist.get_rank() == 0:
            log.info(state.metrics)
    engine.add_event_handler(Events.EPOCH_COMPLETED, function)