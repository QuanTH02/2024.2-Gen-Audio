# Vietnamese Automatic Speech Recognition

This project implements an Automatic Speech Recognition (ASR) system for Vietnamese using the Wav2Vec2 model architecture. The system is built following MLOps best practices and includes features for training, evaluation, and model deployment.

## Features

- Wav2Vec2-based ASR model for Vietnamese
- MLOps pipeline with Hydra for configuration management
- Weights & Biases integration for experiment tracking
- Comprehensive metrics (WER, CER, Accuracy)
- Checkpoint saving and model evaluation
- Support for custom datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vietnamese-asr.git
cd vietnamese-asr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The model is trained on the VLSP2020 dataset and evaluated on the VIVOS dataset. Make sure to download and place the datasets in the appropriate directories:

- Training data: `vlsp2020_train_set_02/`
- Test data: `vivos/`

## Configuration

The project uses Hydra for configuration management. The main configuration file is located at `config/config.yaml`. You can modify the following parameters:

- Model architecture and hyperparameters
- Training settings
- Data processing parameters
- Logging and evaluation settings

## Training

To start training:

```bash
python main.py
```

The training process will:
1. Load and preprocess the dataset
2. Initialize the Wav2Vec2 model
3. Train the model with the specified configuration
4. Log metrics to Weights & Biases
5. Save checkpoints periodically

## Evaluation

The model is evaluated during training using the following metrics:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Accuracy

## Model Architecture

The system uses the Wav2Vec2 model architecture, which is pre-trained on a large corpus of speech data and fine-tuned for Vietnamese ASR. The model includes:

- Feature extraction from raw audio
- Transformer-based encoder
- CTC decoder for transcription

## MLOps Features

- Configuration management with Hydra
- Experiment tracking with Weights & Biases
- Checkpoint saving and model versioning
- Comprehensive logging and monitoring
- Reproducible training pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VLSP2020 dataset
- VIVOS dataset
- Hugging Face Transformers library
- Weights & Biases for experiment tracking 
# 2024.2-Gen-Audio
