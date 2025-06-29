#!/usr/bin/env python3
from __future__ import annotations
import itertools
import math
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.webdataset import make_train_loader, make_val_loader
from models.codec import CodecCfg, ViTCompressorImproved, PerceptualRateDistortionLoss


def lr_schedule(step: int, total_steps: int, base_lr: float, warmup_steps: int):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sweep over λ if requested ---------------------------------------------
    lam_list = cfg.model.lam_list if cfg.model.lam_list else [cfg.model.lam]

    for lam in lam_list:
        print(f"\n=== Training λ = {lam} ===")
        cfg_run = cfg.copy()
        cfg_run.model.lam = lam
        run(cfg_run, device)


def run(cfg: DictConfig, device: str):
    """Improved training function with proper gradient accumulation and logging."""
    
    # Data loaders -----------------------------------------------------------
    train_loader = make_train_loader(cfg)
    val_loader = make_val_loader(cfg)
    
    # Handle infinite vs finite loaders
    try:
        steps_per_epoch = len(train_loader)
        print(f"[info] Found {steps_per_epoch} batches per epoch")
    except TypeError:
        steps_per_epoch = cfg.trainer.get("steps_per_epoch", 1000)
        print(f"[warn] Infinite loader detected → using steps_per_epoch={steps_per_epoch}")
    
    # Calculate total training steps (BEFORE gradient accumulation)
    # LR scheduler should see every mini-batch, not just optimizer steps
    total_steps = cfg.trainer.epochs * steps_per_epoch
    warmup_steps = cfg.trainer.warmup_epochs * steps_per_epoch
    
    print(f"[info] Total training: {total_steps} mini-batch steps, accum_steps={cfg.data.accum_steps}")
    print(f"[info] Optimizer will step every {cfg.data.accum_steps} mini-batches")
    
    # Model + optimizer ------------------------------------------------------
    model_cfg = CodecCfg(**cfg.model)
    model = ViTCompressorImproved(model_cfg).to(device)
    criterion = PerceptualRateDistortionLoss(model_cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] Model: {trainable_params:,} trainable / {total_params:,} total parameters")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.trainer.lr, 
        weight_decay=cfg.trainer.weight_decay
    )
    scaler = GradScaler(enabled=cfg.trainer.amp)
    
    # Training state
    minibatch_step = 0  # Counts every mini-batch
    optimizer_step = 0  # Counts only when optimizer steps
    best_val_loss = float('inf')
    
    # Training loop ----------------------------------------------------------
    for epoch in range(1, cfg.trainer.epochs + 1):
        model.train()
        epoch_loss = 0.0
        accumulated_loss = 0.0
        
        print(f"\n=== Epoch {epoch}/{cfg.trainer.epochs} ===")
        
        for batch_idx, (x,) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=cfg.trainer.amp):
                x_hat, likelihoods = model(x)
                loss, logs = criterion(x_hat, x, likelihoods)
                # Scale loss for gradient accumulation
                scaled_loss = loss / cfg.data.accum_steps
            
            # Backward pass
            scaler.scale(scaled_loss).backward()
            accumulated_loss += scaled_loss.item()
            minibatch_step += 1
            
            # Learning rate scheduling (every mini-batch)
            lr = lr_schedule(minibatch_step, total_steps, cfg.trainer.lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Optimizer step every accum_steps batches OR at end of epoch
            should_step = (batch_idx % cfg.data.accum_steps == 0) or (batch_idx == steps_per_epoch)
            
            if should_step:
                # Gradient clipping and optimizer step
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    cfg.trainer.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Logging
                actual_loss = accumulated_loss * cfg.data.accum_steps
                epoch_loss += actual_loss
                optimizer_step += 1
                
                # Print logs based on optimizer steps
                if optimizer_step % cfg.trainer.log_every == 0:
                    log_dict = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'opt_step': optimizer_step,
                        'loss': round(actual_loss, 4),
                        'bpp': round(logs['bpp'].item(), 4),
                        'mse': round(logs['mse'].item(), 6),
                        'ms_ssim': round(logs['ms_ssim'].item(), 4),
                        'lpips': round(logs['lpips'].item(), 4),
                        'lr': f"{lr:.2e}",
                        'grad_norm': round(grad_norm.item(), 3) if grad_norm is not None else 0.0
                    }
                    print(f"[train] {log_dict}")
                
                # Reset accumulation
                accumulated_loss = 0.0
            
            # Break if using steps_per_epoch limit
            if batch_idx >= steps_per_epoch:
                break
        
        # Epoch summary
        if optimizer_step > 0:
            avg_epoch_loss = epoch_loss / (optimizer_step - (epoch-1) * (steps_per_epoch // cfg.data.accum_steps))
            print(f"[epoch {epoch}] Average loss: {avg_epoch_loss:.4f}, optimizer steps: {optimizer_step}")
        else:
            print(f"[epoch {epoch}] No optimizer steps completed")
        
        # Validation and checkpointing ---------------------------------------
        if epoch % cfg.trainer.save_freq == 0:
            print(f"[info] Running validation...")
            val_metrics = validate(model, val_loader, criterion, device, cfg.trainer.amp)
            val_loss = val_metrics['loss']
            
            # Save checkpoint
            ckpt_dir = Path(cfg.trainer.ckpt_dir)
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': optimizer_step,
                'cfg': model_cfg,
                'val_metrics': val_metrics,
                'train_loss': avg_epoch_loss
            }
            
            # Always save latest
            ckpt_path = ckpt_dir / f"lam{cfg.model.lam}_e{epoch}.pt"
            torch.save(checkpoint, ckpt_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = ckpt_dir / f"lam{cfg.model.lam}_best.pt"
                torch.save(checkpoint, best_path)
                print(f"[info] New best model saved! Val loss: {val_loss:.4f}")
            
            # Format validation metrics for printing
            val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"[val] {val_str}")
            
            # Memory cleanup
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    print(f"\n[info] Training completed! Best validation loss: {best_val_loss:.4f}")


def validate(model, val_loader, criterion, device: str, use_amp: bool = True):
    """Improved validation function with proper error handling."""
    model.eval()
    
    all_metrics = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(val_loader):
            try:
                x = x.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp):
                    x_hat, likelihoods = model(x)
                    _, logs = criterion(x_hat, x, likelihoods)
                
                # Convert to CPU and store
                batch_metrics = {k: v.cpu().item() for k, v in logs.items()}
                all_metrics.append(batch_metrics)
                total_samples += x.size(0)
                
                # Limit validation batches for speed (optional)
                if batch_idx >= 100:  # Validate on first 100 batches max
                    break
                    
            except Exception as e:
                print(f"[warn] Validation batch {batch_idx} failed: {e}")
                continue
    
    if not all_metrics:
        print("[error] No valid validation batches!")
        return {'loss': float('inf'), 'bpp': 0, 'mse': 1, 'ms_ssim': 0, 'lpips': 1}
    
    # Compute averages
    averaged_metrics = {}
    for key in all_metrics[0].keys():
        averaged_metrics[key] = sum(batch[key] for batch in all_metrics) / len(all_metrics)
    
    print(f"[info] Validated on {len(all_metrics)} batches ({total_samples} samples)")
    return averaged_metrics