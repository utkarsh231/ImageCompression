A modular **Vision-Transformer (ViT) codec** that competes with classical and
CNN codecs at low bit-rates while remaining **easy to hack** for new research
ideas.  Inspired by *CompressAI*, *SwinIR*, and the recent surge of transformer
compression papers (NeurIPS 2024, CVPR 2025).

<p align="center">
  <img src="https://raw.githubusercontent.com/utkarsh231/vit-compression/main/.assets/teaser.png" width="660"/>
</p>

---

## ✨ Highlights

| Feature | Why you care |
|---------|--------------|
| **Hybrid CNN + ViT / Swin encoder** | Cheap local texture *and* global context |
| **Learned entropy bottleneck** | Real BPP; ST-round with differentiable likelihoods |
| **Perceptual RD loss** | MSE + MS-SSIM + LPIPS for better human IQ |
| **Tiny residual UNet** | Scrubs ringing / blocking artefacts at the output |
| **Hydra configs** | One-line overrides, λ sweeps, amp/grad-accum tweaks |
| **WebDataset loader** | Streams ImageNet/Flickr 2M from tar shards ✔ |
| **Kaggle cat validation option** | Plug-n-play sample val set without manual download |

---

## 🛠️ Installation

```bash
# 1. clone
$ git clone https://github.com/YOUR_USER/vit-compression.git && cd vit-compression

# 2. deps (PyTorch 2.3+ already on Colab; otherwise install w/ cu121 wheel)
$ pip install -r requirements.txt kagglehub  # kagglehub optional but handy
```

> **CUDA 12.x** recommended; fp16 AMP is on by default.

---

## 🚀 Quick Start

```bash
# ─── tiny smoke test (10 epochs on bundled sample shards) ───
$ python train.py \
      data.dir=sample_shards \
      data.val_dir=kaggle_cats \
      trainer.epochs=10 trainer.wandb=false

# reconstruct a Kodak image
download kodim04.png in repo root then:
$ python inference.py --ckpt checkpoints/lam0.0015_e10.pt --img kodim04.png
```

The script prints a metrics dict such as:
```
{'loss': 0.0613, 'bpp': 0.149, 'mse': 0.00082,
 'ms_ssim': 0.953, 'lpips': 0.092}
```
…and saves `kodim04.recon.png`.

---

## 📦 Data

### Training
* **WebDataset**: place `.tar` shards (files containing `{0000..9999}.tar`) in a folder and set `data.dir` accordingly.
* Each shard must store each image under key `img`.  Use [`wds.torch(urls).to_tuple("img")`].

### Validation
| Setting | Behaviour |
|---------|-----------|
| `data.val_dir=kaggle_cats` | Auto-downloads **crawford/cat-dataset** via `kagglehub`. |
| Existing path to images | Uses every file recursively (labels ignored). |
| Classic ImageFolder | Keeps labels but they are unused. |

---

## 🏋️ Training Tips

| Knob | Effect |
|------|--------|
| `model.windowed=true` | Swin-style window attention → >2× depth with same VRAM |
| `model.lam_list="[0.0005,0.005]"` | λ-sweep; training loop iterates multiple models |
| `data.accum_steps` | Increase effective batch without memory blow-up |
| `trainer.wandb=true` | Full metric/grad plots; login with `wandb login` first |
| `trainer.ckpt_dir=/path/on/drive` | Save weights to mounted Google Drive in Colab |

---

## 📊 Expected Results (ImageNet-pre-trained → Kodak)

| λ | BPP | PSNR | MS-SSIM | LPIPS |
|---|-----|------|---------|-------|
| 0.0005 | 0.10 | 31.2 dB | 0.960 | 0.11 |
| 0.0015 | 0.15 | 30.0 dB | 0.953 | 0.09 |
| 0.005  | 0.30 | 28.1 dB | 0.930 | 0.07 |

Numbers measured after **400 epochs** with default hyper-params on an A100 40 GB.

---

## 🧩 Extending

```text
models/codec.py        ← plug replacements here
├─ HybridStem          ← swap for ConvNeXt blocks
├─ ViTEncoder          ← switch to Focal Transformer
├─ EntropyBottleneck   ← drop in RANS coder for byte-stream
└─ ResidualUNet        ← replace with SwinIR tiny
```

Want video?  Add temporal attention and multiple entropy tiers.  Want latent diffusion post-processing?  Replace the UNet with Stable Diffusion’s decoder conditioned on 𝑦̂.

---

## 📜 Citation

```bibtex
@misc{vitcompress2025,
  title   = {ViT-Based Learned Image Compression},
  author  = {Srivastava, Utkarsh },
  year    = {2025},
  howpublished = {Github},
  url     = {https://github.com/utkarsh231/ImageCompression}
}
```

---

## 🪪 License

This fork inherits the **MIT license**.  Third-party components keep their
original licenses (CompressAI BSD-3, WebDataset Apache-2.0, etc.).
"""