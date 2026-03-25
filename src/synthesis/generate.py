"""
Stable Diffusion generation for Tiny ImageNet and CIFAR-100.
Spec: SD v1.5, DPM-Solver++, 25 steps, guidance 7.5, 512 gen then downscale.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

_IMG_PNG_RE = re.compile(r"^img_(\d+)\.png$", re.IGNORECASE)


def next_synthetic_write_index(class_dir: Path) -> int:
    """
    Next index for img_XXXX.png under class_dir (0 if empty).
    Prefers max numbered img_*.png + 1; falls back to PNG count if none match.
    """
    best = -1
    for p in class_dir.glob("*.png"):
        m = _IMG_PNG_RE.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    if best >= 0:
        return best + 1
    return len(list(class_dir.glob("*.png")))


def cifar100_synthetic_cache_complete(
    output_dir: Path, images_per_class: int, num_classes: int = 100
) -> bool:
    """True when every class folder has at least images_per_class synthetic PNGs."""
    for i in range(num_classes):
        d = output_dir / f"{i:03d}"
        if not d.is_dir():
            return False
        if next_synthetic_write_index(d) < images_per_class:
            return False
    return True

PROMPT_TEMPLATES = [
    "A photo of a {label}",
    "A realistic photo of a {label}",
    "An image depicting a {label}",
    "A clear photograph of a {label}",
]


def _load_wnid_to_labels(root: Path) -> Dict[str, str]:
    words_path = root / "words.txt"
    mapping: Dict[str, str] = {}
    if words_path.exists():
        with words_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1].split(",")[0].strip()
    return mapping


def _get_tiny_class_ids(root: Path) -> List[str]:
    train_dir = root / "train"
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def generate_tiny_imagenet_synthetic(
    output_dir: Path,
    tiny_imagenet_root: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    images_per_class: int = 375,
    batch_size: int = 4,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    image_size: int = 512,
    seed: int = 42,
    device: str = "cuda",
    resume: bool = True,
    post_resize: int = 64,
) -> Path:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    class_ids = _get_tiny_class_ids(tiny_imagenet_root)
    wnid_to_label = _load_wnid_to_labels(tiny_imagenet_root)
    prompts: Dict[str, List[str]] = {}
    for cid in class_ids:
        label = wnid_to_label.get(cid, cid)
        prompts[cid] = [t.format(label=label) for t in PROMPT_TEMPLATES]

    negative_prompt = (
        "blurry, low quality, distorted, watermark, text, "
        "cartoon, drawing, painting, sketch"
    )

    print(f"Loading {model_id} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed)

    log: Dict = {"batches": [], "seed": seed}
    t0 = time.time()
    total_gen, total_skip = 0, 0

    for cid in tqdm(class_ids, desc="Tiny-ImageNet classes"):
        class_dir = output_dir / cid
        class_dir.mkdir(exist_ok=True)
        start_idx = next_synthetic_write_index(class_dir) if resume else 0
        if resume and start_idx >= images_per_class:
            total_skip += images_per_class
            continue

        remaining = images_per_class - start_idx
        class_prompts = prompts[cid]

        for batch_start in range(0, remaining, batch_size):
            actual_bs = min(batch_size, remaining - batch_start)
            batch_prompts = [
                class_prompts[(start_idx + batch_start + j) % len(class_prompts)]
                for j in range(actual_bs)
            ]
            b_t0 = time.time()
            with torch.no_grad():
                result = pipe(
                    batch_prompts,
                    negative_prompt=[negative_prompt] * actual_bs,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=image_size,
                    width=image_size,
                    generator=generator,
                )
            log["batches"].append(
                {"class": cid, "batch_time_s": time.time() - b_t0, "n": actual_bs}
            )

            for j, img in enumerate(result.images):
                idx = start_idx + batch_start + j
                if post_resize and post_resize != image_size:
                    img = img.resize((post_resize, post_resize), Image.Resampling.LANCZOS)
                img.save(class_dir / f"img_{idx:04d}.png")

            total_gen += actual_bs

    elapsed = time.time() - t0
    log["wall_time_s"] = elapsed
    log["images_generated"] = total_gen
    log["images_skipped"] = total_skip
    log["throughput_img_per_s"] = total_gen / elapsed if elapsed > 0 else 0.0
    (output_dir / "generation_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nGeneration complete in {elapsed/60:.1f} min")
    print(f"  Generated: {total_gen}  |  Skipped (existing): {total_skip}")
    print(f"  Output:    {output_dir}")
    return output_dir


def generate_cifar100_synthetic(
    output_dir: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    images_per_class: int = 375,
    batch_size: int = 4,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    image_size: int = 512,
    seed: int = 42,
    device: str = "cuda",
    resume: bool = True,
    post_resize: int = 32,
    cifar_root: Optional[Path] = None,
) -> Path:
    """CIFAR-100: folder per class index 000..099 with human-readable prompts."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from torchvision.datasets import CIFAR100

    if cifar_root is None:
        cifar_root = Path("data/raw/cifar100")
    cifar_root = Path(cifar_root)
    cifar_root.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_gen = 0
    if resume:
        class_indices = [
            i
            for i in range(100)
            if next_synthetic_write_index(output_dir / f"{i:03d}") < images_per_class
        ]
        n_done = 100 - len(class_indices)
        total_skip = n_done * images_per_class
        print(
            f"CIFAR-100 synthetic resume: {len(class_indices)} class(es) need images "
            f"({n_done} already complete)."
        )
    else:
        class_indices = list(range(100))
        total_skip = 0

    if not class_indices:
        log_done: Dict = {
            "batches": [],
            "seed": seed,
            "wall_time_s": 0.0,
            "images_generated": 0,
            "images_skipped": total_skip,
            "throughput_img_per_s": 0.0,
            "cache_complete": True,
        }
        (output_dir / "generation_log.json").write_text(
            json.dumps(log_done, indent=2), encoding="utf-8"
        )
        print("CIFAR-100 synthetic cache already complete; skipped loading SD.")
        return output_dir

    ds = CIFAR100(root=cifar_root, train=True, download=True)
    class_names = ds.classes

    negative_prompt = (
        "blurry, low quality, distorted, watermark, text, "
        "cartoon, drawing, painting, sketch"
    )

    print(f"Loading {model_id} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(seed)

    log: Dict = {"batches": [], "seed": seed}
    t0 = time.time()

    for class_idx in tqdm(class_indices, desc="CIFAR-100 classes"):
        label = class_names[class_idx]
        cid = f"{class_idx:03d}"
        class_dir = output_dir / cid
        class_dir.mkdir(exist_ok=True)
        class_prompts = [t.format(label=label) for t in PROMPT_TEMPLATES]

        start_idx = next_synthetic_write_index(class_dir) if resume else 0
        remaining = images_per_class - start_idx

        for batch_start in range(0, remaining, batch_size):
            actual_bs = min(batch_size, remaining - batch_start)
            batch_prompts = [
                class_prompts[(start_idx + batch_start + j) % len(class_prompts)]
                for j in range(actual_bs)
            ]
            b_t0 = time.time()
            with torch.no_grad():
                result = pipe(
                    batch_prompts,
                    negative_prompt=[negative_prompt] * actual_bs,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=image_size,
                    width=image_size,
                    generator=generator,
                )
            log["batches"].append(
                {"class": cid, "batch_time_s": time.time() - b_t0, "n": actual_bs}
            )

            for j, img in enumerate(result.images):
                idx = start_idx + batch_start + j
                if post_resize and post_resize != image_size:
                    img = img.resize((post_resize, post_resize), Image.Resampling.LANCZOS)
                img.save(class_dir / f"img_{idx:04d}.png")

            total_gen += actual_bs

    elapsed = time.time() - t0
    log["wall_time_s"] = elapsed
    log["images_generated"] = total_gen
    log["images_skipped"] = total_skip
    log["throughput_img_per_s"] = total_gen / elapsed if elapsed > 0 else 0.0
    (output_dir / "generation_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nCIFAR-100 generation complete in {elapsed/60:.1f} min")
    print(f"  Output: {output_dir}")
    return output_dir
