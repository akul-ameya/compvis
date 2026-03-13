from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .tiny_imagenet import get_data_root


class SyntheticTinyImageNetDataset(Dataset):
    """
    Simple loader for synthetic Tiny ImageNet-style images.
    Expected directory layout:
      data/synthetic_sd/<class_id>/img_XXXX.png
    where class_id matches Tiny ImageNet wnid.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if root is None:
            root = get_data_root() / "synthetic_sd"
        self.root = Path(root)
        self.transform = transform

        self.samples: List[Tuple[Path, str]] = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                    continue
                self.samples.append((img_path, class_id))

        if class_to_idx is None:
            class_ids = sorted({cid for _, cid in self.samples})
            self.class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_id = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[class_id]
        return img, label, class_id


class CombinedRealSyntheticDataset(Dataset):
    """
    Dataset that mixes real and synthetic samples with a given ratio.
    Assumes both datasets return (image, label, class_id).
    """

    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_dataset: Dataset,
        real_to_synth_ratio: Tuple[int, int] = (1, 2),
    ) -> None:
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.real_ratio, self.synth_ratio = real_to_synth_ratio
        self.total_ratio = self.real_ratio + self.synth_ratio

    def __len__(self) -> int:
        # Length is defined as real dataset size; synthetic oversamples as needed.
        return len(self.real_dataset) * self.total_ratio

    def __getitem__(self, idx: int):
        slot = idx % self.total_ratio
        if slot < self.real_ratio:
            base_idx = idx // self.total_ratio
            return self.real_dataset[base_idx % len(self.real_dataset)]
        base_idx = idx // self.total_ratio
        return self.synthetic_dataset[base_idx % len(self.synthetic_dataset)]

