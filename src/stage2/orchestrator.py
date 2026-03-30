"""
Stage-2 experiment driver: migrations, training jobs, diagnostics, allocation.
Intended to be called from notebooks (single Run All) or scripts.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.allocation.policies import compute_allocations, load_allocation_csv, save_allocation_csv
from src.config import ExperimentConfig, load_experiment_config
from src.data.registry import (
    build_real_train_subset,
    class_ids_in_label_order,
    get_baseline_loaders,
    get_ceiling_loaders,
)
from src.data.synthetic_dataset import (
    RealSyntheticMixDataset,
    SyntheticImageListDataset,
    list_synthetic_paths_from_allocation,
    list_synthetic_paths_uniform,
)
from src.data.transforms import get_train_transform, get_val_transform
from src.evaluation.fid_stage2 import compute_global_fid_for_ratios
from src.evaluation.stage2_eval import evaluate_stage2
from src.migration.paths import get_resolved_synthetic_root, link_legacy_checkpoints, migrate_tiny_synthetic_layout
from src.models.backbone import build_backbone
from src.stage2.diagnostics import (
    collect_val_predictions,
    compute_class_diagnostics,
    merge_synthetic_quality,
    utility_from_accs,
)
from src.synthesis.generate import (
    cifar100_synthetic_cache_complete,
    generate_cifar100_synthetic,
)
from src.training.stage2_train import train_pipeline


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Stage2Orchestrator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def config_path(self, name: str) -> Path:
        return self.project_root / "configs" / name

    def load_cfg(self, yaml_name: str) -> ExperimentConfig:
        return load_experiment_config(self.config_path(yaml_name), self.project_root)

    # —— migrations ——
    def migrate_tiny_synthetic(self) -> Path:
        cfg = self.load_cfg("tiny_imagenet.yaml")
        migrate_tiny_synthetic_layout(cfg)
        return get_resolved_synthetic_root(cfg)

    def link_stage1_checkpoints(self) -> None:
        cfg = self.load_cfg("tiny_imagenet.yaml")
        link_legacy_checkpoints(cfg)

    def ensure_cifar100_synthetic(self, force: bool = False) -> Path:
        cfg = self.load_cfg("cifar100.yaml")
        cfg.path_synthetic.mkdir(parents=True, exist_ok=True)
        if not force and cifar100_synthetic_cache_complete(
            cfg.path_synthetic, cfg.generation.max_images_per_class
        ):
            return cfg.path_synthetic
        generate_cifar100_synthetic(
            output_dir=cfg.path_synthetic,
            model_id=cfg.generation.model_id,
            images_per_class=cfg.generation.max_images_per_class,
            batch_size=cfg.generation.batch_size,
            num_inference_steps=cfg.generation.inference_steps,
            guidance_scale=cfg.generation.guidance_scale,
            image_size=cfg.generation.image_size,
            seed=cfg.generation.seed,
            device=str(self.device),
            resume=True,
            post_resize=32,
            cifar_root=cfg.project_root / "data" / "raw" / "cifar100",
        )
        return cfg.path_synthetic

    def _run_dir(self, cfg: ExperimentConfig, pipeline: str, arch: str) -> Path:
        d = cfg.path_results_root / cfg.dataset.name / pipeline / arch / _now_tag()
        d.mkdir(parents=True, exist_ok=True)
        snap = {
            "dataset": cfg.dataset.name,
            "pipeline": pipeline,
            "arch": arch,
            "paths": {
                "raw": str(cfg.path_raw),
                "subset": str(cfg.path_subset_index),
                "synthetic": str(cfg.path_synthetic),
            },
        }
        with (d / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        return d

    def _train_job(
        self,
        cfg: ExperimentConfig,
        pipeline: str,
        arch: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_to_idx: Dict[str, int],
        baseline_ckpt_same_arch: Optional[Path] = None,
        real_ds_for_centroids: Optional[object] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        run_dir = self._run_dir(cfg, pipeline, arch)
        model = build_backbone(arch, cfg.dataset.num_classes)

        synth_pipe = pipeline.startswith("uniform") or pipeline.startswith("adaptive")
        bpath = Path(baseline_ckpt_same_arch) if baseline_ckpt_same_arch else None
        use_sa = (
            cfg.training.synthetic_aware_loss
            and synth_pipe
            and bpath is not None
            and bpath.is_file()
            and real_ds_for_centroids is not None
        )
        frozen_ref = None
        centroid_loader = None
        if use_sa:
            frozen_ref = build_backbone(arch, cfg.dataset.num_classes)
            frozen_ref.load_state_dict(torch.load(bpath, map_location=self.device))
            frozen_ref.to(self.device).eval()
            for p in frozen_ref.parameters():
                p.requires_grad_(False)
            centroid_loader = DataLoader(
                real_ds_for_centroids,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        train_pipeline(
            model,
            train_loader,
            val_loader,
            self.device,
            run_dir,
            epochs=cfg.training.epochs,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            mixed_precision=cfg.training.mixed_precision,
            early_stopping_patience=cfg.training.early_stopping_patience,
            synthetic_aware=use_sa,
            frozen_reference_model=frozen_ref,
            centroid_loader=centroid_loader,
            num_classes=cfg.dataset.num_classes,
            synthetic_lambda=cfg.allocation.synthetic_loss_lambda,
        )
        sd = torch.load(run_dir / "best.pt", map_location=self.device)
        model.load_state_dict(sd)

        ref_cka = None
        if bpath is not None and bpath.is_file() and pipeline != "baseline":
            ref_cka = build_backbone(arch, cfg.dataset.num_classes)
            ref_cka.load_state_dict(torch.load(bpath, map_location=self.device))
            ref_cka.to(self.device).eval()

        metrics = evaluate_stage2(
            model,
            val_loader,
            cfg,
            class_to_idx,
            self.device,
            run_dir=run_dir,
            ref_model_for_cka=ref_cka,
        )
        return run_dir, metrics

    def train_baseline(self, cfg_yaml: str, arch: str) -> Tuple[Path, Dict[str, Any]]:
        cfg = self.load_cfg(cfg_yaml)
        cfg.path_figures.mkdir(parents=True, exist_ok=True)
        cfg.path_results_root.mkdir(parents=True, exist_ok=True)
        tr_t = get_train_transform(cfg.dataset.image_size)
        va_t = get_val_transform(cfg.dataset.image_size)
        train_loader, val_loader, c2i = get_baseline_loaders(cfg, tr_t, va_t)
        return self._train_job(cfg, "baseline", arch, train_loader, val_loader, c2i)

    def train_ceiling(
        self,
        cfg_yaml: str,
        arch: str,
        baseline_ckpt_same_arch: Optional[Path] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        cfg = self.load_cfg(cfg_yaml)
        tr_t = get_train_transform(cfg.dataset.image_size)
        va_t = get_val_transform(cfg.dataset.image_size)
        train_loader, val_loader, c2i = get_ceiling_loaders(cfg, tr_t, va_t)
        return self._train_job(
            cfg,
            "ceiling",
            arch,
            train_loader,
            val_loader,
            c2i,
            baseline_ckpt_same_arch=baseline_ckpt_same_arch,
            real_ds_for_centroids=None,
        )

    def train_uniform(
        self,
        cfg_yaml: str,
        arch: str,
        ratio: int,
        baseline_ckpt_same_arch: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        cfg = self.load_cfg(cfg_yaml)
        tr_t = get_train_transform(cfg.dataset.image_size)
        va_t = get_val_transform(cfg.dataset.image_size)
        _, val_loader, c2i = get_baseline_loaders(cfg, tr_t, va_t)
        real_ds, c2i2 = build_real_train_subset(cfg, tr_t)
        assert c2i == c2i2
        synth_root = get_resolved_synthetic_root(cfg)
        k = cfg.synthetic_scaling.per_class_at_ratio[str(ratio)]
        cids = class_ids_in_label_order(c2i)
        items = list_synthetic_paths_uniform(synth_root, cids, k, seed=cfg.generation.seed)
        synth_ds = SyntheticImageListDataset(items, tr_t, c2i)
        b_ok = bool(
            baseline_ckpt_same_arch and Path(baseline_ckpt_same_arch).is_file()
        )
        use_mix = cfg.training.synthetic_aware_loss and b_ok
        full = (
            RealSyntheticMixDataset(real_ds, synth_ds)
            if use_mix
            else ConcatDataset([real_ds, synth_ds])
        )
        train_loader = DataLoader(
            full,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        pipeline = name or f"uniform_{ratio}x"
        return self._train_job(
            cfg,
            pipeline,
            arch,
            train_loader,
            val_loader,
            c2i,
            baseline_ckpt_same_arch=baseline_ckpt_same_arch,
            real_ds_for_centroids=real_ds if use_mix else None,
        )

    def train_adaptive(
        self,
        cfg_yaml: str,
        arch: str,
        allocation_csv: Path,
        name: str = "adaptive_15x",
        baseline_ckpt_same_arch: Optional[Path] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        cfg = self.load_cfg(cfg_yaml)
        tr_t = get_train_transform(cfg.dataset.image_size)
        va_t = get_val_transform(cfg.dataset.image_size)
        _, val_loader, c2i = get_baseline_loaders(cfg, tr_t, va_t)
        real_ds, c2i2 = build_real_train_subset(cfg, tr_t)
        assert c2i == c2i2
        synth_root = get_resolved_synthetic_root(cfg)
        alloc = load_allocation_csv(Path(allocation_csv))
        items = list_synthetic_paths_from_allocation(synth_root, alloc, seed=cfg.generation.seed)
        synth_ds = SyntheticImageListDataset(items, tr_t, c2i)
        b_ok = bool(
            baseline_ckpt_same_arch and Path(baseline_ckpt_same_arch).is_file()
        )
        use_mix = cfg.training.synthetic_aware_loss and b_ok
        full = (
            RealSyntheticMixDataset(real_ds, synth_ds)
            if use_mix
            else ConcatDataset([real_ds, synth_ds])
        )
        train_loader = DataLoader(
            full,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return self._train_job(
            cfg,
            name,
            arch,
            train_loader,
            val_loader,
            c2i,
            baseline_ckpt_same_arch=baseline_ckpt_same_arch,
            real_ds_for_centroids=real_ds if use_mix else None,
        )

    def compute_baseline_diagnostics(
        self,
        cfg_yaml: str,
        checkpoint: Path,
        arch: str = "resnet18",
        quality_csv: Optional[Path] = None,
    ) -> Path:
        cfg = self.load_cfg(cfg_yaml)
        tr_t = get_train_transform(cfg.dataset.image_size)
        va_t = get_val_transform(cfg.dataset.image_size)
        _, val_loader, _ = get_baseline_loaders(cfg, tr_t, va_t)
        model = build_backbone(arch, cfg.dataset.num_classes)
        model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        model.to(self.device)
        logits, labels, feats = collect_val_predictions(
            model, val_loader, self.device, cfg.dataset.num_classes
        )
        _, _, c2i = get_baseline_loaders(cfg, tr_t, va_t)
        cids = class_ids_in_label_order(c2i)
        df = compute_class_diagnostics(logits, labels, feats, cids)
        df = merge_synthetic_quality(df, quality_csv)
        out = cfg.path_results_root / cfg.dataset.name / "diagnostics" / arch
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "class_diagnostics.csv"
        df.to_csv(csv_path, index_label="class_id")
        return csv_path

    def build_allocations(
        self,
        cfg_yaml: str,
        diagnostics_csv: Path,
        utility_json: Optional[Path],
        policies: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        cfg = self.load_cfg(cfg_yaml)
        tr_t = get_train_transform(cfg.dataset.image_size)
        va_t = get_val_transform(cfg.dataset.image_size)
        _, _, c2i = get_baseline_loaders(cfg, tr_t, va_t)
        cids = class_ids_in_label_order(c2i)

        df = pd.read_csv(diagnostics_csv)
        if "class_id" in df.columns:
            df = df.set_index("class_id")
        df.index = df.index.astype(str)
        df = df.reindex(cids).fillna(0.0)
        num_c = len(cids)
        total_budget = num_c * cfg.allocation.max_cap
        util = None
        if utility_json and Path(utility_json).exists():
            util = json.loads(Path(utility_json).read_text(encoding="utf-8"))
        pols = policies or ["uniform", "hard_class", "uncertainty", "predicted_utility"]
        out_dir = cfg.path_results_root / cfg.dataset.name / "allocations"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for pol in pols:
            if pol == "predicted_utility" and util is None:
                continue
            alloc, meta = compute_allocations(
                pol,
                cids,
                total_budget,
                cfg.allocation.min_floor,
                cfg.allocation.max_cap,
                df,
                utility=util,
                cv_folds=cfg.allocation.cv_folds,
            )
            p = out_dir / f"allocation_{pol}.csv"
            save_allocation_csv(alloc, p)
            paths[pol] = p
            if meta:
                with (out_dir / f"allocation_{pol}_meta.json").open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
        return paths

    def utility_from_metrics(
        self,
        baseline_metrics: Dict[str, Any],
        uniform_metrics: Dict[str, Any],
        class_ids_ordered: List[str],
    ) -> Dict[str, float]:
        ab = {int(k): v for k, v in baseline_metrics["per_class_acc"].items()}
        au = {int(k): v for k, v in uniform_metrics["per_class_acc"].items()}
        return utility_from_accs(ab, au, class_ids_ordered)

    def compute_global_fid(
        self,
        cfg_yaml: str,
        ratios: Optional[List[int]] = None,
        force_resync_images: bool = False,
    ) -> Dict[str, Any]:
        """Global FID (5% real vs subsampled synthetic); requires `pip install clean-fid`."""
        cfg = self.load_cfg(cfg_yaml)
        return compute_global_fid_for_ratios(
            cfg, ratios=ratios, force_resync_images=force_resync_images
        )

    def aggregate_results_index(self, cfg_yaml: str) -> Path:
        cfg = self.load_cfg(cfg_yaml)
        root = cfg.path_results_root / cfg.dataset.name
        rows = []
        if not root.exists():
            return root / "index.json"
        for pipeline_dir in root.iterdir():
            if not pipeline_dir.is_dir() or pipeline_dir.name in ("diagnostics", "allocations", "legacy"):
                continue
            for arch_dir in pipeline_dir.iterdir():
                if not arch_dir.is_dir():
                    continue
                for run in arch_dir.iterdir():
                    if not run.is_dir():
                        continue
                    mp = run / "metrics.json"
                    if mp.exists():
                        m = json.loads(mp.read_text(encoding="utf-8"))
                        rows.append(
                            {
                                "pipeline": pipeline_dir.name,
                                "arch": arch_dir.name,
                                "run": run.name,
                                "top1": m.get("top1"),
                                "path": str(run),
                            }
                        )
        idx = root / "results_index.json"
        idx.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return idx
