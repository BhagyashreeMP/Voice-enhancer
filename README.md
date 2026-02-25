# Speech Enhancement — Fine-Tuning Project

Fine-tune **Facebook's pre-trained Demucs** model for speech enhancement on your custom audio data.

## Why Fine-Tuning?

| | Training from Scratch | Fine-Tuning (this project) |
|---|---|---|
| **Data needed** | 10,000+ paired samples | 50–500 paired samples |
| **Training time** | 100+ epochs | 20–30 epochs |
| **Pre-trained weights** | None (random init) | Facebook Demucs DNS48 |
| **Parameters** | ~116M (DCCRN) | ~18M (Demucs) |
| **Result quality** | Depends on data volume | Good even with small data |

## Pre-Trained Model: Facebook Demucs

- **Architecture:** Encoder-Decoder with skip connections + LSTM bottleneck
- **Variant:** DNS48 (~18M params) — trained on Microsoft DNS Challenge (500+ hours)
- **Paper:** [Real Time Speech Enhancement in the Waveform Domain](https://arxiv.org/abs/2006.12847)
- **Library:** [`denoiser`](https://github.com/facebookresearch/denoiser)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r finetune/requirements.txt
```

### 2. Prepare Data

Place your paired audio data in this structure:
```
data/finetune/
  train/
    clean/  *.wav
    noisy/  *.wav   (matching filenames)
  val/
    clean/  *.wav
    noisy/  *.wav
```

Or generate synthetic data for testing:
```bash
python finetune/dataset.py
```

### 3. Fine-Tune

```bash
# Full fine-tune (all layers)
python finetune/finetune.py

# Fine-tune decoder only (freeze encoder)
python finetune/finetune.py --freeze-encoder

# Generate data + fine-tune
python finetune/finetune.py --generate-data

# Resume training
python finetune/finetune.py --resume finetune/checkpoints/epoch_0010.pth
```

### 4. Evaluate

```bash
# Evaluate fine-tuned model
python finetune/evaluate.py

# Compare fine-tuned vs pre-trained baseline
python finetune/evaluate.py --compare
```

### 5. Run Streamlit App

```bash
streamlit run finetune/app.py
```

## Project Structure

```
finetune/
  config.yaml       — Fine-tuning configuration
  dataset.py        — Dataset & synthetic data generator
  finetune.py       — Fine-tuning training loop
  evaluate.py       — Evaluation & comparison script
  app.py            — Streamlit web app
  requirements.txt  — Python dependencies
  README.md         — This file
  checkpoints/      — Saved model checkpoints (created during training)
  logs/             — Training logs & TensorBoard
```

## Configuration

Key settings in `config.yaml`:

```yaml
pretrained:
  model_name: facebook/demucs-dns48   # Pre-trained model
  freeze_encoder: false                # Freeze encoder layers

training:
  epochs: 30                          # Fewer than from-scratch
  learning_rate: 0.0001               # 10x lower than from-scratch
  batch_size: 8
```

## Alternative Pre-Trained Models

| Model | Library | Params | Best For |
|---|---|---|---|
| **Demucs DNS48** ✓ | `denoiser` | 18M | General speech enhancement |
| Demucs DNS64 | `denoiser` | 33M | Higher quality, more compute |
| Asteroid DCCRNet | `asteroid` | 3.7M | Low-latency, CPU deployment |
| SpeechBrain MetricGAN+ | `speechbrain` | 1.8M | Perceptual quality (PESQ) |

To switch models, update `pretrained.model_name` in `config.yaml`.
