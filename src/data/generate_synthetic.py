"""
Generate synthetic Tiny ImageNet images using Stable Diffusion.

Usage (standalone):
    python -m src.data.generate_synthetic --images-per-class 375

Usage (from notebook / script):
    from src.data.generate_synthetic import generate_synthetic_images
    generate_synthetic_images(output_dir, images_per_class=375)
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


def load_wnid_to_labels(root: Path) -> Dict[str, str]:
    words_path = root / "words.txt"
    mapping = {}
    if words_path.exists():
        with words_path.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1].split(",")[0].strip()
    return mapping


def get_class_ids(root: Path) -> List[str]:
    train_dir = root / "train"
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


PROMPT_TEMPLATES = [
    "a photo of a {label}, high quality, detailed, realistic",
    "a photograph of a {label}, natural lighting, sharp focus",
    "a clear image of a {label}, professional photography",
    "a {label}, realistic, well-lit, high resolution",
]


def build_prompts(
    class_ids: List[str],
    wnid_to_label: Dict[str, str],
) -> Dict[str, List[str]]:
    prompts = {}
    for cid in class_ids:
        label = wnid_to_label.get(cid, cid)
        prompts[cid] = [t.format(label=label) for t in PROMPT_TEMPLATES]
    return prompts


def generate_synthetic_images(
    output_dir: Path,
    tiny_imagenet_root: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    images_per_class: int = 200,
    batch_size: int = 4,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    image_size: int = 512,
    seed: int = 42,
    device: str = "cuda",
    resume: bool = True,
) -> Path:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    class_ids = get_class_ids(tiny_imagenet_root)
    wnid_to_label = load_wnid_to_labels(tiny_imagenet_root)
    prompts = build_prompts(class_ids, wnid_to_label)

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

    total_gen, total_skip = 0, 0
    t0 = time.time()

    for cid in tqdm(class_ids, desc="Classes"):
        class_dir = output_dir / cid
        class_dir.mkdir(exist_ok=True)
        existing = len(list(class_dir.glob("*.png")))
        if resume and existing >= images_per_class:
            total_skip += images_per_class
            continue

        start_idx = existing if resume else 0
        remaining = images_per_class - start_idx
        class_prompts = prompts[cid]

        for batch_start in range(0, remaining, batch_size):
            actual_bs = min(batch_size, remaining - batch_start)
            batch_prompts = [
                class_prompts[(start_idx + batch_start + j) % len(class_prompts)]
                for j in range(actual_bs)
            ]

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

            for j, img in enumerate(result.images):
                idx = start_idx + batch_start + j
                img.save(class_dir / f"img_{idx:04d}.png")

            total_gen += actual_bs

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed/60:.1f} min")
    print(f"  Generated: {total_gen}  |  Skipped (existing): {total_skip}")
    print(f"  Output:    {output_dir}")
    return output_dir


if __name__ == "__main__":
    from .tiny_imagenet import get_data_root, get_tiny_imagenet_root

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--images-per-class", type=int, default=375)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else get_data_root() / "synthetic_sd"

    generate_synthetic_images(
        output_dir=out,
        tiny_imagenet_root=get_tiny_imagenet_root(),
        model_id=args.model_id,
        images_per_class=args.images_per_class,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        seed=args.seed,
        resume=not args.no_resume,
    )
