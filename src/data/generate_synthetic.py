"""
CLI / module entry for synthetic image generation.
Tiny ImageNet: delegates to src.synthesis.generate (512 → 64).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.synthesis.generate import generate_tiny_imagenet_synthetic


def main() -> None:
    from src.data.tiny_imagenet import get_data_root, get_tiny_imagenet_root

    parser = argparse.ArgumentParser(description="Generate synthetic Tiny ImageNet images (SD v1.5)")
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
    generate_tiny_imagenet_synthetic(
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
        post_resize=64,
    )


if __name__ == "__main__":
    main()
