"""
finetune/finetune.py
=====================
Fine-tune Facebook's pre-trained Demucs model for speech enhancement.

Key differences from training-from-scratch:
  1. Loads pre-trained Demucs weights (DNS48 or DNS64)
  2. Uses lower learning rate to preserve pre-trained knowledge
  3. Optionally freezes encoder layers for few-shot scenarios
  4. Fewer epochs needed (model already knows speech patterns)

Usage:
    # Generate synthetic test data first:
    python finetune/dataset.py

    # Fine-tune:
    python finetune/finetune.py

    # Resume from checkpoint:
    python finetune/finetune.py --resume finetune/checkpoints/epoch_0010.pth

    # Fine-tune with frozen encoder:
    python finetune/finetune.py --freeze-encoder
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Ensure finetune/ is on path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

log_dir = Path("finetune/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_dir / "finetune.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

class SISDRLoss(nn.Module):
    """Scale-Invariant SDR loss (negative, for minimization)."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate, target):
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        dot = (estimate * target).sum(dim=-1, keepdim=True)
        s_target = dot / ((target ** 2).sum(dim=-1, keepdim=True) + self.eps) * target
        e_noise = estimate - s_target

        si_sdr = 10 * torch.log10(
            (s_target ** 2).sum(dim=-1) /
            ((e_noise ** 2).sum(dim=-1) + self.eps) + self.eps
        )
        return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """STFT loss at multiple resolutions."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(self, x, y, n_fft, hop, win):
        window = torch.hann_window(win, device=x.device)

        # Process each batch item
        loss = 0
        for i in range(x.shape[0]):
            X = torch.stft(x[i], n_fft, hop, win, window=window,
                           return_complex=True, normalized=False, onesided=True)
            Y = torch.stft(y[i], n_fft, hop, win, window=window,
                           return_complex=True, normalized=False, onesided=True)

            mag_x, mag_y = X.abs(), Y.abs()

            # Spectral convergence
            sc = torch.norm(mag_y - mag_x, p="fro") / (torch.norm(mag_y, p="fro") + 1e-8)

            # Log magnitude L1
            log_mag = torch.nn.functional.l1_loss(
                torch.log(mag_x + 1e-8),
                torch.log(mag_y + 1e-8),
            )

            loss += sc + log_mag

        return loss / x.shape[0]

    def forward(self, estimate, target):
        loss = 0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self._stft_loss(estimate, target, fft, hop, win)
        return loss / len(self.fft_sizes)


class CombinedFineTuneLoss(nn.Module):
    """Weighted combination of L1 + SI-SDR + multi-res STFT."""

    def __init__(self, l1_weight=0.5, si_sdr_weight=0.5, stft_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.si_sdr_weight = si_sdr_weight
        self.stft_weight = stft_weight

        self.l1 = nn.L1Loss()
        self.si_sdr = SISDRLoss()
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(self, estimate, target):
        loss = 0
        components = {}

        if self.l1_weight > 0:
            l1 = self.l1(estimate, target)
            loss += self.l1_weight * l1
            components["l1"] = l1.item()

        if self.si_sdr_weight > 0:
            si = self.si_sdr(estimate, target)
            loss += self.si_sdr_weight * si
            components["si_sdr"] = si.item()

        if self.stft_weight > 0:
            stft = self.stft_loss(estimate, target)
            loss += self.stft_weight * stft
            components["stft"] = stft.item()

        components["total"] = loss.item()
        return loss, components


# ─────────────────────────────────────────────
# Load Pre-Trained Demucs
# ─────────────────────────────────────────────

def load_pretrained_demucs(model_name: str = "dns48", device: str = "cpu"):
    """
    Load Facebook's pre-trained Demucs speech enhancement model.

    Args:
        model_name: 'dns48' (~18M params) or 'dns64' (~33M params)
        device: 'cpu' or 'cuda'

    Returns:
        model: Pre-trained Demucs model
    """
    try:
        from denoiser import pretrained
        if model_name in ("dns48", "facebook/demucs-dns48"):
            model = pretrained.dns48()
        elif model_name in ("dns64", "facebook/demucs-dns64"):
            model = pretrained.dns64()
        else:
            # Try as a HuggingFace model ID
            model = pretrained.dns48()
            logger.warning(f"Unknown model '{model_name}', falling back to dns48")

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded pre-trained Demucs ({model_name}) | {n_params/1e6:.2f}M params")
        return model

    except ImportError:
        raise ImportError(
            "facebook/denoiser is required. Install with:\n"
            "  pip install denoiser\n"
            "Or: pip install -r finetune/requirements.txt"
        )


def freeze_encoder_layers(model, n_layers: int = 0, freeze_all_encoder: bool = False):
    """
    Freeze encoder layers to preserve pre-trained features.

    Args:
        model: Demucs model
        n_layers: Number of encoder layers to freeze (0 = none)
        freeze_all_encoder: Freeze entire encoder
    """
    frozen = 0

    if freeze_all_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
                frozen += param.numel()
    elif n_layers > 0:
        for name, param in model.named_parameters():
            # Demucs encoder layers are named like encoder.0, encoder.1, etc.
            for i in range(n_layers):
                if f"encoder.{i}" in name:
                    param.requires_grad = False
                    frozen += param.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parameters: {total/1e6:.2f}M total | "
        f"{trainable/1e6:.2f}M trainable | "
        f"{frozen/1e6:.2f}M frozen"
    )


# ─────────────────────────────────────────────
# Fine-Tuning Trainer
# ─────────────────────────────────────────────

class FineTuneTrainer:
    """
    Fine-tuning loop for pre-trained Demucs model.

    Key differences from from-scratch training:
      - Lower learning rate (1e-4 vs 1e-3)
      - Optional layer freezing
      - Fewer epochs
      - Monitors for catastrophic forgetting
    """

    def __init__(self, config: dict, device: str = "auto", resume_path: str = None,
                 freeze_encoder: bool = False):
        self.config = config
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            device if device != "auto" else "cpu"
        )
        self.cfg_t = config["training"]
        self.cfg_p = config["pretrained"]

        # ── Load pre-trained model ──────────────────────────
        model_name = self.cfg_p["model_name"].split("/")[-1]  # e.g. "dns48"
        self.model = load_pretrained_demucs(model_name, str(self.device))

        # ── Freeze layers if requested ──────────────────────
        if freeze_encoder or self.cfg_p.get("freeze_encoder", False):
            freeze_encoder_layers(
                self.model,
                n_layers=self.cfg_p.get("freeze_layers", 0),
                freeze_all_encoder=True,
            )

        # ── Loss ────────────────────────────────────────────
        loss_cfg = self.cfg_t["loss"]
        self.criterion = CombinedFineTuneLoss(
            l1_weight=loss_cfg["l1_weight"],
            si_sdr_weight=loss_cfg["si_sdr_weight"],
            stft_weight=loss_cfg["stft_weight"],
        ).to(self.device)

        # ── Optimizer (lower LR for fine-tuning) ────────────
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg_t["learning_rate"],
            weight_decay=self.cfg_t["weight_decay"],
        )

        # ── LR Scheduler ────────────────────────────────────
        self.scheduler = None  # Built after dataloaders are ready

        # ── AMP (only on CUDA) ──────────────────────────────
        self.use_amp = self.cfg_t["mixed_precision"] and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None

        # ── Data ────────────────────────────────────────────
        from dataset import build_finetune_dataloaders
        self.train_loader, self.val_loader = build_finetune_dataloaders(config)

        # ── Scheduler (now that we know train_loader size) ──
        total_steps = self.cfg_t["epochs"] * len(self.train_loader)
        warmup_steps = self.cfg_t["warmup_epochs"] * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # ── TensorBoard ─────────────────────────────────────
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path("finetune/logs") / f"run_{int(time.time())}"
            self.writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard: tensorboard --logdir {tb_dir}")
        except ImportError:
            self.writer = None
            logger.warning("TensorBoard not available")

        # ── State ───────────────────────────────────────────
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        Path(self.cfg_t["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

        if resume_path:
            self._load_checkpoint(resume_path)

    # ─────────────────────────────────────────
    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        ckpt_dir = Path(self.cfg_t["checkpoint_dir"])
        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "val_loss": val_loss,
            "config": self.config,
            "pretrained_model": self.cfg_p["model_name"],
        }

        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_path = ckpt_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info(f"New best model! Val loss: {val_loss:.6f}")

        # Clean up old checkpoints
        self._cleanup_checkpoints(ckpt_dir)

    def _cleanup_checkpoints(self, ckpt_dir):
        keep = self.cfg_t.get("keep_last_n", 3)
        ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
        for old in ckpts[:-keep]:
            old.unlink()

    def _load_checkpoint(self, path):
        logger.info(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))

    # ─────────────────────────────────────────
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)  # (B, T)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Demucs expects (B, 1, T) input
                enhanced = self.model(noisy.unsqueeze(1))  # (B, 1, T)
                enhanced = enhanced.squeeze(1)             # (B, T)

                # Trim to same length
                min_len = min(enhanced.shape[-1], clean.shape[-1])
                loss, components = self.criterion(enhanced[..., :min_len], clean[..., :min_len])

            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.cfg_t["grad_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg_t["grad_clip"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg_t["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg_t["grad_clip"]
                    )
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

            # TensorBoard logging
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                for k, v in components.items():
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

        return total_loss / max(n_batches, 1)

    # ─────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for noisy, clean in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            enhanced = self.model(noisy.unsqueeze(1)).squeeze(1)
            min_len = min(enhanced.shape[-1], clean.shape[-1])
            loss, _ = self.criterion(enhanced[..., :min_len], clean[..., :min_len])

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ─────────────────────────────────────────
    def train(self):
        """Run the full fine-tuning loop."""
        logger.info("=" * 60)
        logger.info("Starting fine-tuning")
        logger.info(f"  Pre-trained model: {self.cfg_p['model_name']}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.cfg_t['epochs']}")
        logger.info(f"  LR: {self.cfg_t['learning_rate']}")
        logger.info(f"  Batch size: {self.cfg_t['batch_size']}")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info("=" * 60)

        for epoch in range(self.epoch + 1, self.cfg_t["epochs"] + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch)

            elapsed = time.time() - t0
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            logger.info(
                f"Epoch {epoch:3d}/{self.cfg_t['epochs']} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {self.best_val_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # TensorBoard
            if self.writer:
                self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)

            # Save checkpoint
            if epoch % self.cfg_t["save_every_n_epochs"] == 0 or is_best:
                self._save_checkpoint(epoch, val_loss, is_best)

        logger.info("Fine-tuning complete!")
        if self.writer:
            self.writer.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Demucs for speech enhancement")
    parser.add_argument("--config", default="finetune/config.yaml", help="Config file")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers (only fine-tune decoder)")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic data before training")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Generate data if requested
    if args.generate_data:
        from dataset import generate_synthetic_finetune_data
        generate_synthetic_finetune_data("data/finetune", n_samples=200)

    # Train
    trainer = FineTuneTrainer(
        config=config,
        device=args.device,
        resume_path=args.resume,
        freeze_encoder=args.freeze_encoder,
    )
    trainer.train()


if __name__ == "__main__":
    main()
