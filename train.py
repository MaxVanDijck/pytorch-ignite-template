import logging

import hydra
import ignite.distributed as idist
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(config: DictConfig):
    from src.train_pipeline import train

    log.info(f"Starting run with Backend: {config.compute.backend}")
    log.info(f"Num Processes per Node: {config.compute.nproc_per_node}")

    with idist.launcher.Parallel(
        backend=config.compute.backend, 
        nproc_per_node=config.compute.nproc_per_node
        ) as parallel:
        parallel.run(train, config)

if __name__ == "__main__":
    main()