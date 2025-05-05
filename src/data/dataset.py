import os
import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from typing import Dict, List, Union
import librosa
import soundfile as sf
import random

class VietnameseASRDataset:
    def __init__(self, data_path: str, config: Dict, is_test: bool = False, train_frac: float = 1.0):
        self.data_path = data_path
        self.config = config
        self.sample_rate = config["data"]["sample_rate"]
        self.max_duration = config["data"]["max_duration"]
        self.min_duration = config["data"]["min_duration"]
        self.is_test = is_test
        self.train_frac = train_frac

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze()

    def prepare_dataset(self) -> Dataset:
        """Prepare dataset from raw data."""
        data = []
        total_files = 0
        matched_files = 0
        duration_filtered = 0
        error_files = 0

        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    total_files += 1
                    audio_path = os.path.join(root, file)
                    transcript_path = audio_path.replace('.wav', '.txt').replace('.mp3', '.txt')

                    if not os.path.exists(transcript_path):
                        print(f"[SKIP] Không tìm thấy transcript cho: {audio_path}")
                        continue

                    try:
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                    except Exception as e:
                        print(f"[ERROR] Lỗi đọc transcript {transcript_path}: {e}")
                        error_files += 1
                        continue

                    try:
                        waveform = self.load_audio(audio_path)
                    except Exception as e:
                        print(f"[ERROR] Lỗi đọc audio {audio_path}: {e}")
                        error_files += 1
                        continue

                    duration = waveform.shape[0] / self.sample_rate

                    if not (self.min_duration <= duration <= self.max_duration):
                        print(f"[SKIP] Duration {duration:.2f}s ngoài khoảng [{self.min_duration}, {self.max_duration}] cho: {audio_path}")
                        duration_filtered += 1
                        continue

                    matched_files += 1
                    data.append({
                        "audio": waveform,
                        "text": transcript,
                        "duration": duration
                    })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Set random seed for reproducibility
        random.seed(42)

        if self.is_test:
            df = df.sample(frac=0.05, random_state=42)
        else:
            df = df.sample(frac=self.train_frac, random_state=42)

        print(f"[{self.data_path}] Tổng file audio: {total_files}, matched: {matched_files}, lỗi: {error_files}, bị loại do duration: {duration_filtered}, còn lại sau lọc: {len(df)}")

        return Dataset.from_pandas(df)

    def collate_fn(self, batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        input_values = [item["audio"] for item in batch]
        labels = [item["text"] for item in batch]
        
        # Pad audio sequences
        input_values = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        
        return {
            "input_values": input_values,
            "labels": labels
        } 