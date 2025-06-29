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
    # data loaders -----------------------------------------------------------
    train_loader = make_train_loader(cfg)
    val_loader = make_val_loader(cfg)

    try:
        steps_per_epoch = len(train_loader)               # ImageFolder etc.
    except TypeError:
        steps_per_epoch = cfg.trainer.get("steps_per_epoch", 1000)
        print(f"[warn] infinite loader → using steps_per_epoch={steps_per_epoch}")

    total_steps  = cfg.trainer.epochs       * steps_per_epoch
    warmup_steps = cfg.trainer.warmup_epochs * steps_per_epoch

    # model + optimiser ------------------------------------------------------
    model_cfg = CodecCfg(**cfg.model)
    model = ViTCompressorImproved(model_cfg).to(device)
    criterion = PerceptualRateDistortionLoss(model_cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    scaler = GradScaler(enabled=cfg.trainer.amp)

    step = 0
    accum = cfg.data.accum_steps
    for epoch in range(1, cfg.trainer.epochs + 1):
        model.train()
        for i, (x,) in enumerate(train_loader, 1):
            x = x.to(device)
            with autocast(enabled=cfg.trainer.amp):
                x_hat, likelihoods = model(x)
                loss, logs = criterion(x_hat, x, likelihoods)
                loss = loss / accum
            scaler.scale(loss).backward()

            if i % accum == 0:
                # grad clipping & step -------------------------------------
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            # LR schedule --------------------------------------------------
            lr = lr_schedule(step, total_steps, cfg.trainer.lr, warmup_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr

            if step % cfg.trainer.log_every == 0:
                print({k: round(v.item(), 4) for k, v in logs.items()} | {"lr": lr})
            step += 1

        # validation ---------------------------------------------------------
        if epoch % cfg.trainer.save_freq == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            ckpt_dir = Path(cfg.trainer.ckpt_dir)
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'cfg': model_cfg,
            }, ckpt_dir / f"lam{cfg.model.lam}_e{epoch}.pt")
            print(f"[val] {val_metrics}")


def validate(model, loader: DataLoader, criterion, device):
    model.eval()
    with torch.no_grad():
        metrics = []
        for (x,) in loader:
            x = x.to(device)
            x_hat, likelihoods = model(x)
            _, logs = criterion(x_hat, x, likelihoods)
            metrics.append({k: v.item() for k, v in logs.items()})
        # average
        agg = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    return agg


if __name__ == "__main__":
    main()