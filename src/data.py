"""Dataset loading for 140k Real and Fake Faces."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def find_root(data_dir):
    """Find the folder that holds the train/valid/test splits, wherever the zip put it."""
    data_dir = Path(data_dir)
    for train_dir in data_dir.rglob("train"):
        root = train_dir.parent
        if (root / "valid").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError(
        f"No train/valid/test folders under {data_dir}. Run src/download_data.py first."
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(train):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def make_loader(data_dir, split, batch_size=64, num_workers=4, subset=None, seed=0):
    """Return (loader, class_names) for split in {'train', 'valid', 'test'}.

    Classes come from the folder names in sorted order, so fake=0 and real=1.
    subset limits the number of images, for quick experiments.
    """
    root = find_root(data_dir)
    train = split == "train"
    dataset = datasets.ImageFolder(root / split, transform=build_transforms(train))

    if subset and subset < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:subset].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ["fake", "real"]
