#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torchvision import transforms as T
from PIL import Image

from models.codec import CodecCfg, ViTCompressorImproved, PerceptualRateDistortionLoss


def main():
    p = argparse.ArgumentParser(description="Compress & reconstruct a single image")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--img", type=str, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt['cfg'] if 'cfg' in ckpt else CodecCfg()

    model = ViTCompressorImproved(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tfm = T.Compose([T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()])
    x = tfm(Image.open(args.img).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        x_hat, likelihoods = model(x)
        criterion = PerceptualRateDistortionLoss(cfg)
        _, logs = criterion(x_hat, x, likelihoods)
        print({k: round(v.item(),4) for k,v in logs.items()})

    out = Path(args.img).with_suffix(".recon.png")
    T.ToPILImage()(x_hat.squeeze().cpu()).save(out)
    print(f"saved → {out}")

if __name__ == "__main__":
    main()