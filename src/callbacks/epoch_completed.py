import logging

log = logging.getLogger(__name__)

def log_lr(engine):
    log.info(engine.state.optimizer.param_groups[0]["lr"])
