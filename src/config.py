"""
YAML-backed experiment configuration (dataclass hierarchy).
Load with load_experiment_config(yaml_path, project_root).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str  # tiny_imagenet | cifar100
    num_classes: int = 200
    real_images_per_class: int = 25
    image_size: int = 224


@dataclass
class PathsConfig:
    raw_data: str = ""
    subset_index: str = ""
    synthetic_root: str = ""
    synthetic_legacy: str = ""
    results_root: str = "results"
    figures_stage2: str = "figures/stage2"
    checkpoints_legacy: str = "checkpoints"


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    mixed_precision: bool = True
    early_stopping_patience: int = 7
    scheduler: str = "cosine"
    # If True, uniform/adaptive jobs use distance-to-centroid weights (needs baseline checkpoint).
    synthetic_aware_loss: bool = False


@dataclass
class GenerationConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    max_images_per_class: int = 375
    batch_size: int = 4
    inference_steps: int = 25
    guidance_scale: float = 7.5
    image_size: int = 512
    seed: int = 42


@dataclass
class SyntheticScalingConfig:
    ratios: List[int] = field(default_factory=lambda: [5, 10, 15])
    per_class_at_ratio: Dict[str, int] = field(default_factory=dict)


@dataclass
class AllocationConfig:
    min_floor: int = 50
    max_cap: int = 375
    cv_folds: int = 10
    synthetic_loss_lambda: float = 5.0


@dataclass
class MetricsConfig:
    ece_bins: int = 15
    worst_k_classes: int = 20
    cka_subsample: int = 2000
    linear_probe_epochs: int = 20
    linear_probe_lr: float = 0.01


@dataclass
class ScopeConfig:
    tiny_architectures: List[str] = field(default_factory=lambda: ["resnet18", "mobilenet_v3_small"])
    cifar_architectures: List[str] = field(default_factory=lambda: ["resnet18"])
    cifar_pipelines: List[str] = field(
        default_factory=lambda: ["baseline", "uniform_15x", "adaptive_15x", "ceiling"]
    )
    adaptive_policies_tiny: List[str] = field(
        default_factory=lambda: ["hard_class", "uncertainty", "predicted_utility"]
    )
    # Which allocation CSV to use for CIFAR adaptive 15× (filename stem after allocation_)
    cifar_adaptive_policy: str = "predicted_utility"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    paths: PathsConfig
    training: TrainingConfig
    generation: GenerationConfig
    synthetic_scaling: SyntheticScalingConfig
    allocation: AllocationConfig
    metrics: MetricsConfig
    scope: ScopeConfig = field(default_factory=ScopeConfig)

    project_root: Path = field(default_factory=Path)
    path_raw: Path = field(default_factory=Path)
    path_subset_index: Path = field(default_factory=Path)
    path_synthetic: Path = field(default_factory=Path)
    path_results_root: Path = field(default_factory=Path)
    path_figures: Path = field(default_factory=Path)


def load_experiment_config(yaml_path: Path, project_root: Optional[Path] = None) -> ExperimentConfig:
    yaml_path = Path(yaml_path).resolve()
    if project_root is None:
        project_root = yaml_path.parent.parent
    project_root = Path(project_root).resolve()

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    d = raw.get("dataset", {})
    dataset = DatasetConfig(
        name=d.get("name", "tiny_imagenet"),
        num_classes=int(d.get("num_classes", 200)),
        real_images_per_class=int(d.get("real_images_per_class", 25)),
        image_size=int(d.get("image_size", 224)),
    )
    p = raw.get("paths", {})
    paths = PathsConfig(
        raw_data=p.get("raw_data", ""),
        subset_index=p.get("subset_index", ""),
        synthetic_root=p.get("synthetic_root", ""),
        synthetic_legacy=p.get("synthetic_legacy", ""),
        results_root=p.get("results_root", "results"),
        figures_stage2=p.get("figures_stage2", "figures/stage2"),
        checkpoints_legacy=p.get("checkpoints_legacy", "checkpoints"),
    )
    t = raw.get("training", {})
    training = TrainingConfig(
        epochs=int(t.get("epochs", 30)),
        batch_size=int(t.get("batch_size", 64)),
        num_workers=int(t.get("num_workers", 0)),
        learning_rate=float(t.get("learning_rate", 3e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        mixed_precision=bool(t.get("mixed_precision", True)),
        early_stopping_patience=int(t.get("early_stopping_patience", 7)),
        scheduler=str(t.get("scheduler", "cosine")),
        synthetic_aware_loss=bool(t.get("synthetic_aware_loss", False)),
    )
    g = raw.get("generation", {})
    generation = GenerationConfig(
        model_id=g.get("model_id", "runwayml/stable-diffusion-v1-5"),
        max_images_per_class=int(g.get("max_images_per_class", 375)),
        batch_size=int(g.get("batch_size", 4)),
        inference_steps=int(g.get("inference_steps", 25)),
        guidance_scale=float(g.get("guidance_scale", 7.5)),
        image_size=int(g.get("image_size", 512)),
        seed=int(g.get("seed", 42)),
    )
    ss = raw.get("synthetic_scaling", {})
    ratios = ss.get("ratios", [5, 10, 15])
    ratios = [int(x) for x in ratios]
    pcar = ss.get("per_class_at_ratio")
    if pcar:
        per_class = {str(k): int(v) for k, v in pcar.items()}
    else:
        per_class = {str(r): r * dataset.real_images_per_class for r in ratios}
    synthetic_scaling = SyntheticScalingConfig(ratios=ratios, per_class_at_ratio=per_class)

    a = raw.get("allocation", {})
    allocation = AllocationConfig(
        min_floor=int(a.get("min_floor", 50)),
        max_cap=int(a.get("max_cap", 375)),
        cv_folds=int(a.get("cv_folds", 10)),
        synthetic_loss_lambda=float(a.get("synthetic_loss_lambda", 5.0)),
    )
    m = raw.get("metrics", {})
    metrics = MetricsConfig(
        ece_bins=int(m.get("ece_bins", 15)),
        worst_k_classes=int(m.get("worst_k_classes", 20)),
        cka_subsample=int(m.get("cka_subsample", 2000)),
        linear_probe_epochs=int(m.get("linear_probe_epochs", 20)),
        linear_probe_lr=float(m.get("linear_probe_lr", 0.01)),
    )
    sc = raw.get("scope", {})
    scope = ScopeConfig(
        tiny_architectures=list(sc.get("tiny_architectures", ["resnet18", "mobilenet_v3_small"])),
        cifar_architectures=list(sc.get("cifar_architectures", ["resnet18"])),
        cifar_pipelines=list(
            sc.get("cifar_pipelines", ["baseline", "uniform_15x", "adaptive_15x", "ceiling"])
        ),
        adaptive_policies_tiny=list(
            sc.get("adaptive_policies_tiny", ["hard_class", "uncertainty", "predicted_utility"])
        ),
        cifar_adaptive_policy=str(sc.get("cifar_adaptive_policy", "predicted_utility")),
    )

    cfg = ExperimentConfig(
        dataset=dataset,
        paths=paths,
        training=training,
        generation=generation,
        synthetic_scaling=synthetic_scaling,
        allocation=allocation,
        metrics=metrics,
        scope=scope,
    )
    cfg.project_root = project_root
    cfg.path_raw = (project_root / paths.raw_data).resolve()
    cfg.path_subset_index = (project_root / paths.subset_index).resolve()
    cfg.path_synthetic = (project_root / paths.synthetic_root).resolve()
    cfg.path_results_root = (project_root / paths.results_root).resolve()
    cfg.path_figures = (project_root / paths.figures_stage2).resolve()

    return cfg
