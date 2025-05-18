import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Optional

class VietnameseASRModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load pretrained model and processor
        self.model = Wav2Vec2ForCTC.from_pretrained(
            config["model"]["name"],
            output_hidden_states=True
        )
        
        self.processor = Wav2Vec2Processor.from_pretrained(
            config["model"]["name"]
        )
        
        # Freeze feature extractor if specified
        if config["model"]["freeze_feature_extractor"]:
            self.model.freeze_feature_extractor()
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        input_values = batch["input_values"]
        
        # Process audio inputs
        input_values = self.processor(
            input_values,
            sampling_rate=self.config["data"]["sample_rate"],
            return_tensors="pt"
        ).input_values
        
        return {
            "input_values": input_values,
            "labels": batch.get("labels")
        }
    
    def decode(self, logits: torch.Tensor) -> str:
        """Decode model outputs to text."""
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0] 