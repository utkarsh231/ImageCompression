"""codec.py – fully corrected core model + loss for the ViT‑based image
compression project.  Copy into `models/codec.py`.
"""
from __future__ import annotations

import argparse, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from piq import ms_ssim, LPIPS

# ─────────────────────────────────────────────────────────────── CONFIG ─────
@dataclass
class CodecCfg:
    img_size: int = 256
    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 8
    heads: int = 6
    windowed: bool = False

    use_hybrid_stem: bool = True
    latent_channels: int = 384
    use_gaussian_conditional: bool = False

    lam: float = 0.0015
    alpha_ms_ssim: float = 0.75
    beta_mse: float = 0.25
    gamma_lpips: float = 0.05

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    lam_list: List[float] = field(default_factory=lambda: [0.0005, 0.0015, 0.005, 0.015])

# ────────────────────────────────────────────────────────────── MODULES ─────
class HybridStem(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1), nn.GELU(),
        )
    def forward(self, x: Tensor) -> Tensor:  # (B,3,H,W) → (B,p,H/4,W/4)
        return self.net(x)

class ResidualUNet(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(ch, 2*ch, 3, 2, 1), nn.GELU())
        self.mid   = nn.Sequential(nn.Conv2d(2*ch, 2*ch, 3, 1, 1), nn.GELU())
        self.up2   = nn.Sequential(nn.ConvTranspose2d(2*ch, ch, 2, 2), nn.GELU())
        self.out   = nn.Conv2d(ch, 3, 3, 1, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m  = self.mid(d2)
        u  = self.up2(m) + d1
        return self.out(u)

# ───────────────────────────────────────────────────────────── MAIN MODEL ───
class ViTCompressorImproved(nn.Module):
    def __init__(self, cfg: CodecCfg):
        super().__init__()
        self.cfg = cfg
        
        # Cleaner approach - stick to standard patch sizes
        if cfg.use_hybrid_stem:
            self.stem = HybridStem(cfg.embed_dim)  # Output embed_dim channels
            encoder_in_chans = cfg.embed_dim
            img_size_for_vit = cfg.img_size // 4  # Due to 2 strided convs
        else:
            self.stem = nn.Identity()
            encoder_in_chans = 3
            img_size_for_vit = cfg.img_size
            
        # Standard ViT
        self.encoder = VisionTransformer(
            img_size=img_size_for_vit,
            patch_size=cfg.patch_size,
            in_chans=encoder_in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.heads,
            class_token=False,  # Explicitly disable CLS token
            num_classes=0,      # No classification head
        )
        # entropy
        self.entropy_model = GaussianConditional(None) if cfg.use_gaussian_conditional else EntropyBottleneck(cfg.latent_channels)
        # decoder ---------------------------------------------------
        self.full_patch = cfg.patch_size * (4 if cfg.use_hybrid_stem else 1)  # size in the *original* image grid
        self.decoder = nn.Linear(cfg.embed_dim, self.full_patch * self.full_patch * 3)
        self.post_net = ResidualUNet()

    # helpers
    def _bchw(self, t, h, w): return t.transpose(1,2).contiguous().view(t.size(0), t.size(2), h, w)
    def _tokens(self, y): B, D, H, W = y.shape; return y.view(B, D, H*W).transpose(1,2).contiguous()

    def forward(self, x: Tensor):
        B, _, H0, W0 = x.shape
        x_s = self.stem(x)
        tokens = self.encoder.forward_features(x_s) if hasattr(self.encoder,'forward_features') else self.encoder(x_s)
        # Calculate patch grid size after stem
        vit_img = H0 // 4 if self.cfg.use_hybrid_stem else H0
        htok = vit_img // self.cfg.patch_size
        wtok = vit_img // self.cfg.patch_size
        # Remove class token if present
        if tokens.size(1) == htok * wtok + 1:
            tokens = tokens[:, 1:]
        assert tokens.size(1) == htok * wtok, f"Token count {tokens.size(1)} != {htok}x{wtok}"
        y = self._bchw(tokens, htok, wtok)
        y_hat, lik = self.entropy_model(y)
        tok_hat = self._tokens(y_hat)
        p  = self.full_patch
        N = tok_hat.size(1)
        patches = self.decoder(tok_hat)
        patches = patches.view(B, N, 3, p, p)
        flat = patches.permute(0,2,3,4,1).contiguous().view(B, 3*p*p, N)
        x_hat = F.fold(
            flat,
            output_size=(H0, W0),
            kernel_size=p,
            stride=p,
        )
        x_hat = torch.clamp(x_hat + self.post_net(x_hat), 0., 1.)
        return x_hat, lik

# ───────────────────────────────────────────────────────────── LOSS ────────
class PerceptualRateDistortionLoss(nn.Module):
    def __init__(self,cfg:CodecCfg):
        super().__init__(); self.cfg=cfg; self.lpips=LPIPS(reduction='mean')
    def forward(self,xh,x,lik):
        mse=F.mse_loss(xh,x); mss=1-ms_ssim(xh,x,data_range=1.); lp=self.lpips(xh,x)
        dist=self.cfg.beta_mse*mse+self.cfg.alpha_ms_ssim*mss+self.cfg.gamma_lpips*lp
        bpp=-torch.log2(lik).sum()/(x.numel()/x.size(0))
        loss=dist+self.cfg.lam*bpp
        return loss,{"loss":loss.detach(),"bpp":bpp.detach(),"mse":mse.detach(),"ms_ssim":1-mss.detach(),"lpips":lp.detach()}

# ───────────────────────────────────────────────────────────── CLI DEMO ────
if __name__ == "__main__":
    import argparse, torchvision.transforms as T
    from PIL import Image
    ap=argparse.ArgumentParser(); ap.add_argument("--img",required=True); args=ap.parse_args()
    cfg=CodecCfg(); model=ViTCompressorImproved(cfg).to(cfg.device)
    img=T.Compose([T.Resize((cfg.img_size,cfg.img_size)),T.ToTensor()])(Image.open(args.img).convert("RGB")).unsqueeze(0).to(cfg.device)
    with torch.no_grad(): out,_=model(img)
    T.ToPILImage()(out.squeeze()).save(Path(args.img).with_suffix('.recon.png'))
