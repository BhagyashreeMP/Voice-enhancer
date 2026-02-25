"""
finetune/evaluate.py
=====================
Evaluate the fine-tuned Demucs model against the pre-trained baseline.

Computes:
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - PESQ  (Perceptual Evaluation of Speech Quality)
  - STOI  (Short-Time Objective Intelligibility)

Usage:
    python finetune/evaluate.py --checkpoint finetune/checkpoints/best_model.pth
    python finetune/evaluate.py --compare   # Compare fine-tuned vs pre-trained
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
import soundfile as sf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-Invariant SDR in dB."""
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    dot = np.dot(estimate, target)
    s_target = dot / (np.dot(target, target) + 1e-8) * target
    e_noise = estimate - s_target
    return float(10 * np.log10(
        (np.dot(s_target, s_target) + 1e-8) /
        (np.dot(e_noise, e_noise) + 1e-8)
    ))


def compute_pesq_score(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> Optional[float]:
    try:
        from pesq import pesq
        return float(pesq(sr, clean, enhanced, "wb"))
    except Exception:
        return None


def compute_stoi_score(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> Optional[float]:
    try:
        from pystoi import stoi
        return float(stoi(clean, enhanced, sr, extended=False))
    except Exception:
        return None


def load_finetuned_model(checkpoint_path: str, device: str = "cpu"):
    """Load the fine-tuned Demucs model from checkpoint."""
    from denoiser import pretrained

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model_name = config["pretrained"]["model_name"].split("/")[-1]

    if model_name == "dns64":
        model = pretrained.dns64()
    else:
        model = pretrained.dns48()

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, config


def load_pretrained_baseline(model_name: str = "dns48", device: str = "cpu"):
    """Load the original pre-trained model (for comparison)."""
    from denoiser import pretrained

    if model_name == "dns64":
        model = pretrained.dns64()
    else:
        model = pretrained.dns48()

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def enhance_with_model(model, noisy_wav: np.ndarray, sr: int = 16000,
                       device: str = "cpu") -> np.ndarray:
    """Run enhancement through a Demucs model."""
    wav_t = torch.from_numpy(noisy_wav).float().unsqueeze(0).unsqueeze(0)  # (1,1,T)
    wav_t = wav_t.to(device)
    enhanced = model(wav_t)
    return enhanced.squeeze().cpu().numpy()


def evaluate_on_directory(
    model,
    clean_dir: str,
    noisy_dir: str,
    sr: int = 16000,
    device: str = "cpu",
    max_samples: int = 50,
) -> Dict[str, float]:
    """Evaluate model on a directory of paired files."""
    clean_dir = Path(clean_dir)
    noisy_dir = Path(noisy_dir)

    clean_files = sorted(clean_dir.glob("*.wav"))[:max_samples]

    metrics = {"si_sdr": [], "si_sdr_input": [], "pesq": [], "stoi": []}

    for cf in clean_files:
        nf = noisy_dir / cf.name
        if not nf.exists():
            continue

        clean_np, _ = sf.read(str(cf))
        noisy_np, _ = sf.read(str(nf))

        # Align lengths
        min_len = min(len(clean_np), len(noisy_np))
        clean_np = clean_np[:min_len]
        noisy_np = noisy_np[:min_len]

        # Input quality (before enhancement)
        metrics["si_sdr_input"].append(compute_si_sdr(noisy_np, clean_np))

        # Enhance
        enhanced_np = enhance_with_model(model, noisy_np, sr, device)
        enhanced_np = enhanced_np[:min_len]

        # Output quality (after enhancement)
        metrics["si_sdr"].append(compute_si_sdr(enhanced_np, clean_np))

        pesq_score = compute_pesq_score(clean_np, enhanced_np, sr)
        if pesq_score is not None:
            metrics["pesq"].append(pesq_score)

        stoi_score = compute_stoi_score(clean_np, enhanced_np, sr)
        if stoi_score is not None:
            metrics["stoi"].append(stoi_score)

    results = {}
    for k, v in metrics.items():
        if v:
            results[k] = float(np.mean(v))
            results[f"{k}_std"] = float(np.std(v))
    
    if "si_sdr" in results and "si_sdr_input" in results:
        results["si_sdr_improvement"] = results["si_sdr"] - results["si_sdr_input"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--checkpoint", default="finetune/checkpoints/best_model.pth")
    parser.add_argument("--config", default="finetune/config.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compare", action="store_true",
                        help="Also evaluate pre-trained baseline for comparison")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    clean_dir = config["data"]["val_clean_dir"]
    noisy_dir = config["data"]["val_noisy_dir"]
    sr = config["audio"]["sample_rate"]

    # ── Evaluate fine-tuned model ────────────────────────
    logger.info("Evaluating fine-tuned model...")
    ft_model, _ = load_finetuned_model(args.checkpoint, device)
    ft_results = evaluate_on_directory(ft_model, clean_dir, noisy_dir, sr, device, args.max_samples)

    print("\n" + "=" * 50)
    print("FINE-TUNED MODEL RESULTS")
    print("=" * 50)
    for k, v in ft_results.items():
        print(f"  {k:25s}: {v:.4f}")

    # ── Compare with pre-trained baseline ────────────────
    if args.compare:
        model_name = config["pretrained"]["model_name"].split("/")[-1]
        logger.info(f"\nEvaluating pre-trained baseline ({model_name})...")
        baseline = load_pretrained_baseline(model_name, device)
        bl_results = evaluate_on_directory(baseline, clean_dir, noisy_dir, sr, device, args.max_samples)

        print("\n" + "=" * 50)
        print(f"PRE-TRAINED BASELINE ({model_name})")
        print("=" * 50)
        for k, v in bl_results.items():
            print(f"  {k:25s}: {v:.4f}")

        print("\n" + "=" * 50)
        print("IMPROVEMENT (Fine-tuned - Baseline)")
        print("=" * 50)
        for k in ["si_sdr", "pesq", "stoi"]:
            if k in ft_results and k in bl_results:
                diff = ft_results[k] - bl_results[k]
                sign = "+" if diff > 0 else ""
                print(f"  {k:25s}: {sign}{diff:.4f}")


if __name__ == "__main__":
    main()
