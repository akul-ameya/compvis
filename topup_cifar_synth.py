#!/usr/bin/env python3
"""
Top-up CIFAR-100 synthetic images for class 026 (135 → 375).
Run this BEFORE restarting the notebook.
Usage:  python topup_cifar_synth.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthesis.generate import generate_cifar100_synthetic

print("Topping up CIFAR-100 synthetic images (resume=True) …")
generate_cifar100_synthetic(
    output_dir=PROJECT_ROOT / "data" / "synthetic" / "cifar100",
    model_id="runwayml/stable-diffusion-v1-5",
    images_per_class=375,
    batch_size=4,
    num_inference_steps=25,
    guidance_scale=7.5,
    image_size=512,
    seed=42,
    device="cuda",
    resume=True,
    post_resize=32,
    cifar_root=PROJECT_ROOT / "data" / "raw" / "cifar100",
)
print("✅ Done — all CIFAR-100 classes now have 375 synthetic images.")
