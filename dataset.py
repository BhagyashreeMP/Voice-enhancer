"""
finetune/dataset.py
====================
Dataset for fine-tuning the pre-trained Demucs model.

Loads paired (noisy, clean) WAV files — same format as the
original project but tailored for fine-tuning workflow.
Also includes a synthetic data generator for quick testing.
"""

import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader


class FineTuneDataset(Dataset):
    """
    Paired (noisy, clean) dataset for fine-tuning.

    Expects:
        clean_dir/  *.wav
        noisy_dir/  *.wav   (filenames must match)

    Returns:
        noisy: (T,) float32 tensor
        clean: (T,) float32 tensor
    """

    def __init__(
        self,
        clean_dir: Union[str, Path],
        noisy_dir: Union[str, Path],
        sample_rate: int = 16000,
        segment_duration: float = 4.0,
        augment: bool = False,
        mode: str = "train",
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_len = int(segment_duration * sample_rate)
        self.augment = augment
        self.mode = mode

        self.file_list = self._find_pairs()
        print(f"[FineTuneDataset/{mode}] Found {len(self.file_list)} pairs")

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        clean_files = sorted(self.clean_dir.glob("*.wav"))
        pairs = []
        for cf in clean_files:
            nf = self.noisy_dir / cf.name
            if nf.exists():
                pairs.append((cf, nf))
            else:
                nf2 = self.noisy_dir / (cf.stem + "_noisy.wav")
                if nf2.exists():
                    pairs.append((cf, nf2))
        return pairs

    def __len__(self):
        return len(self.file_list)

    def _load(self, path: Path) -> torch.Tensor:
        data, sr = sf.read(str(path), dtype='float32')
        wav = torch.from_numpy(data).float()
        if wav.dim() == 2:
            wav = wav.mean(dim=-1)  # stereo to mono
        if sr != self.sample_rate:
            from scipy.signal import resample_poly
            import math
            gcd = math.gcd(self.sample_rate, sr)
            up, down = self.sample_rate // gcd, sr // gcd
            resampled = resample_poly(wav.numpy(), up, down)
            wav = torch.from_numpy(resampled).float()
        return wav  # (T,)

    def _crop_or_pad(self, clean: torch.Tensor, noisy: torch.Tensor):
        T = clean.shape[0]
        L = self.segment_len

        if T < L:
            clean = torch.nn.functional.pad(clean, (0, L - T))
            noisy = torch.nn.functional.pad(noisy, (0, L - T))
        elif T > L:
            start = random.randint(0, T - L) if self.mode == "train" else (T - L) // 2
            clean = clean[start:start + L]
            noisy = noisy[start:start + L]
        return clean, noisy

    def _apply_augmentation(self, clean, noisy):
        # Random gain
        gain_db = random.uniform(-6, 6)
        gain = 10 ** (gain_db / 20)
        clean, noisy = clean * gain, noisy * gain

        # Random polarity flip
        if random.random() < 0.5:
            clean, noisy = -clean, -noisy

        return clean, noisy

    def __getitem__(self, idx):
        clean_path, noisy_path = self.file_list[idx]
        clean = self._load(clean_path)
        noisy = self._load(noisy_path)

        # Align lengths
        min_len = min(clean.shape[0], noisy.shape[0])
        clean, noisy = clean[:min_len], noisy[:min_len]

        clean, noisy = self._crop_or_pad(clean, noisy)

        if self.augment and self.mode == "train":
            clean, noisy = self._apply_augmentation(clean, noisy)

        # Peak normalize
        peak = max(clean.abs().max(), noisy.abs().max(), 1e-8)
        clean = clean / peak * 0.9
        noisy = noisy / peak * 0.9

        return noisy, clean


def build_finetune_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from fine-tune config."""
    cfg_d = config["data"]
    cfg_a = config["audio"]
    cfg_t = config["training"]

    train_ds = FineTuneDataset(
        clean_dir=cfg_d["train_clean_dir"],
        noisy_dir=cfg_d["train_noisy_dir"],
        sample_rate=cfg_a["sample_rate"],
        segment_duration=cfg_a["segment_duration"],
        augment=cfg_d["augmentation"]["enabled"],
        mode="train",
    )

    val_ds = FineTuneDataset(
        clean_dir=cfg_d["val_clean_dir"],
        noisy_dir=cfg_d["val_noisy_dir"],
        sample_rate=cfg_a["sample_rate"],
        segment_duration=cfg_a["segment_duration"],
        augment=False,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_t["batch_size"],
        shuffle=True,
        num_workers=cfg_t["num_workers"],
        pin_memory=cfg_t["pin_memory"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_t["batch_size"],
        shuffle=False,
        num_workers=cfg_t["num_workers"],
        pin_memory=cfg_t["pin_memory"],
        drop_last=False,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────
# Synthetic Data Generator (for quick testing)
# ─────────────────────────────────────────────

def generate_synthetic_finetune_data(
    output_dir: Union[str, Path],
    n_samples: int = 200,
    sr: int = 16000,
    duration: float = 4.0,
):
    """
    Generate synthetic paired data for testing the fine-tuning pipeline.

    Creates harmonics-based "clean speech" + Gaussian noise = "noisy".
    Only for pipeline testing — use real data for actual fine-tuning.
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    splits = {
        "train": int(n_samples * 0.8),
        "val": n_samples - int(n_samples * 0.8),
    }

    for split, count in splits.items():
        clean_dir = output_dir / split / "clean"
        noisy_dir = output_dir / split / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)

        n_samples_audio = int(duration * sr)
        t = torch.linspace(0, duration, n_samples_audio)

        for i in range(count):
            freq = random.uniform(80, 300)
            clean = sum(
                torch.sin(2 * torch.pi * freq * k * t) / k
                for k in range(1, 8)
            ).float()

            snr_db = random.uniform(-5, 20)
            noise = torch.randn_like(clean)
            snr_lin = 10 ** (snr_db / 20)
            noise = noise / (noise.std() + 1e-8) * (clean.std() / (snr_lin + 1e-8))
            noisy = clean + noise

            clean = (clean / (clean.abs().max() + 1e-8) * 0.9).numpy()
            noisy = (noisy / (noisy.abs().max() + 1e-8) * 0.9).numpy()

            name = f"sample_{i:05d}.wav"
            sf.write(str(clean_dir / name), clean, sr)
            sf.write(str(noisy_dir / name), noisy, sr)

    print(f"Synthetic fine-tune data saved to {output_dir}")
    print(f"  Train: {splits['train']} pairs | Val: {splits['val']} pairs")


if __name__ == "__main__":
    generate_synthetic_finetune_data("data/finetune", n_samples=200)
