import os
import hydra
from omegaconf import DictConfig
import wandb
from src.training.trainer import ASRTrainer

@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ASRTrainer(config)
    
    # Start training
    trainer.train()
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 