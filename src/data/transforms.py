from typing import Callable, Tuple

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def gaussian_noise_transform(std: float = 0.1) -> Callable:
    def _transform(x):
        return (x + std * x.new_empty(x.size()).normal_()).clamp(0.0, 1.0)

    return _transform


def brightness_shift_transform(factor: float = 0.2) -> Callable:
    def _transform(x):
        return (x + factor).clamp(0.0, 1.0)

    return _transform


def blur_transform(kernel_size: int = 3) -> Callable:
    blur = transforms.GaussianBlur(kernel_size=kernel_size)

    def _transform(x):
        return blur(x)

    return _transform


def apply_corruption(
    base_transform: Callable, corruption: Callable
) -> Callable:
    """
    Compose a normalization-free base transform with a corruption.
    Assumes base_transform returns a tensor in [0, 1].
    """

    def _composed(img):
        x = base_transform(img)
        return corruption(x)

    return _composed

