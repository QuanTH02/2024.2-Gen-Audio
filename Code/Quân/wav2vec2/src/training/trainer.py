import os
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, List

from src.data.dataset import VietnameseASRDataset
from src.models.model import VietnameseASRModel
from src.utils.metrics import calculate_wer, calculate_cer, calculate_accuracy, get_wer_components

class ASRTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )
        
        # Initialize model and dataset
        self.model = VietnameseASRModel(config).to(self.device)
        self.train_dataset = VietnameseASRDataset(
            config.data.train_data_path, 
            config, 
            is_test=False,
            train_frac=0.2
        )
        self.eval_dataset = VietnameseASRDataset(
            config.data.test_data_path, 
            config, 
            is_test=True
        )
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset.prepare_dataset(),
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            collate_fn=self.train_dataset.collate_fn
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset.prepare_dataset(),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=self.eval_dataset.collate_fn
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.model.warmup_steps,
            num_training_steps=config.model.max_steps
        )
    
    def train(self):
        """Training loop."""
        self.model.train()
        global_step = 0
        total_steps = self.config.model.max_steps
        total_epochs = total_steps // len(self.train_loader) + 1

        for epoch in range(total_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch")
            for batch in pbar:
                # Prepare inputs
                inputs = self.model.prepare_inputs(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (global_step + 1) % self.config.model.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if global_step % self.config.training.logging_steps == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/step": global_step
                    })
                    # Log tiến độ ra console
                    pbar.set_postfix({
                        "step": f"{global_step}/{total_steps}",
                        "loss": f"{loss.item():.4f}"
                    })
                
                # Evaluation
                if global_step % self.config.training.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    wandb.log({
                        "eval/wer": eval_metrics["wer"],
                        "eval/cer": eval_metrics["cer"],
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/substitutions": eval_metrics["substitutions"],
                        "eval/deletions": eval_metrics["deletions"],
                        "eval/insertions": eval_metrics["insertions"],
                        "eval/step": global_step
                    })
                
                # Save checkpoint
                if global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1
                
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                inputs = self.model.prepare_inputs(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Decode predictions
                predictions = self.model.decode(logits)
                references = batch["labels"]
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Calculate metrics
        wer = calculate_wer(all_predictions, all_references)
        cer = calculate_cer(all_predictions, all_references)
        accuracy = calculate_accuracy(all_predictions, all_references)
        
        # Get detailed WER components
        wer_components = get_wer_components(all_predictions, all_references)
        
        self.model.train()
        return {
            "wer": wer,
            "cer": cer,
            "accuracy": accuracy,
            **wer_components
        }
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.training.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step
        }, os.path.join(checkpoint_dir, "model.pt")) 