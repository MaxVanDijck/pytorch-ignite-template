from omegaconf import DictConfig
import hydra
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="train")
def main(config: DictConfig):
    from src.train_pipeline import train
    return train(config)

if __name__ == "__main__":
    main()