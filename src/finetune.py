"""Fine-tune the detector on modern diffusion faces, mixed with a sample of the original StyleGAN data.

Run from the project folder:  python src/finetune.py
The result is saved to checkpoints/finetuned.pt; checkpoints/best.pt is left untouched.
"""
import argparse
import io
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data import IMAGE_SIZE, MEAN, STD, find_root, get_device
from model import build_model
from predict import Detector

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CACHE = Path("data_extra/crops")
CROP_SIDE = 256
FAKE, REAL = 0, 1  # same class order as the original training (folder order: fake, real)


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTENSIONS)


def cached_crops(detector, source, name):
    """Face-crop every image in `source` once and return the cached files. Images with no face are skipped."""
    out = CACHE / name
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for path in tqdm(list_images(source), desc=f"crop {name}", leave=False):
        target = out / f"{path.stem}.png"
        if not target.exists():
            face, found = detector.crop_face(Image.open(path).convert("RGB"))
            if not found:
                continue
            face.resize((CROP_SIDE, CROP_SIDE), Image.BICUBIC).save(target)
        files.append(target)
    return files


def jpeg_roundtrip(image):
    """Re-compress as JPEG so file format (PNG vs JPG) can't be used as a shortcut."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=random.randint(50, 95))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class Faces(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            steps += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1)]
        steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label, _group = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.train and random.random() < 0.7:
            image = jpeg_roundtrip(image)
        return self.transform(image), label


def sample(files, count, rng):
    return rng.sample(files, min(count, len(files)))


def evaluate(model, loader, groups, device):
    """Return (balanced accuracy, {group: fraction correct}) over the validation items."""
    model.eval()
    correct = {}
    with torch.no_grad():
        index = 0
        for images, labels in loader:
            predicted = (model(images.to(device)).squeeze(1) > 0).long().cpu()
            for guess, label in zip(predicted, labels):
                correct.setdefault(groups[index], []).append(int(guess == label))
                index += 1
    per_group = {g: sum(v) / len(v) for g, v in correct.items()}
    real = [c for g, v in correct.items() if g.startswith("real") for c in v]
    fake = [c for g, v in correct.items() if not g.startswith("real") for c in v]
    return (sum(real) / len(real) + sum(fake) / len(fake)) / 2, per_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/best.pt", help="model to start from")
    parser.add_argument("--out", default="checkpoints/finetuned.pt")
    parser.add_argument("--new", default="data_extra/diffusion_faces", help="folder with train/val of the new dataset")
    parser.add_argument("--orig-per-class", type=int, default=3000, help="original-dataset images per class")
    parser.add_argument("--gan-per-class", type=int, default=700, help="new-dataset GAN fakes to include")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(0)
    device = get_device()
    print("Device:", device)

    detector = Detector(args.base)  # also used to crop faces in the new dataset
    new = Path(args.new)
    orig = find_root("data")

    new_real = cached_crops(detector, new / "train" / "real", "train_real")
    diffusion = cached_crops(detector, new / "train" / "diffusion", "train_diffusion")
    gan = cached_crops(detector, new / "train" / "gan", "train_gan")
    val_real = cached_crops(detector, new / "val" / "real", "val_real")
    val_diffusion = cached_crops(detector, new / "val" / "diffusion", "val_diffusion")
    val_gan = cached_crops(detector, new / "val" / "gan", "val_gan")

    # Photoshop-edited images are left out: the goal here is AI-generated faces.
    train_items = (
        [(p, REAL, "real_new") for p in new_real] * 2  # modern real photos are scarce, so show them twice
        + [(p, FAKE, "diffusion") for p in diffusion]
        + [(p, FAKE, "gan_new") for p in sample(gan, args.gan_per_class, rng)]
        + [(p, REAL, "real_orig") for p in sample(list_images(orig / "train" / "real"), args.orig_per_class, rng)]
        + [(p, FAKE, "fake_orig") for p in sample(list_images(orig / "train" / "fake"), args.orig_per_class, rng)]
    )
    val_items = (
        [(p, REAL, "real_new") for p in val_real]
        + [(p, FAKE, "diffusion") for p in val_diffusion]
        + [(p, FAKE, "gan_new") for p in val_gan]
        + [(p, REAL, "real_orig") for p in sample(list_images(orig / "valid" / "real"), 500, rng)]
        + [(p, FAKE, "fake_orig") for p in sample(list_images(orig / "valid" / "fake"), 500, rng)]
    )
    print(f"train {len(train_items):,} images, validation {len(val_items):,} images")

    train_loader = DataLoader(Faces(train_items, True), args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(Faces(val_items, False), 64, num_workers=args.workers)
    val_groups = [item[2] for item in val_items]

    state = torch.load(args.base, map_location=device)
    model = build_model(state["model"], pretrained=False).to(device)
    model.load_state_dict(state["state_dict"])
    del detector

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    score, per_group = evaluate(model, val_loader, val_groups, device)
    print(f"Before fine-tuning: balanced acc {score:.3f}  " + "  ".join(f"{g} {v:.2f}" for g, v in per_group.items()))

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            logits = model(images.to(device)).squeeze(1)
            loss = criterion(logits, labels.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)

        score, per_group = evaluate(model, val_loader, val_groups, device)
        print(f"Epoch {epoch}: loss {total / count:.4f}  balanced acc {score:.3f}  "
              + "  ".join(f"{g} {v:.2f}" for g, v in per_group.items()))
        if score > best:
            best = score
            Path(args.out).parent.mkdir(exist_ok=True)
            torch.save({"model": state["model"], "state_dict": model.state_dict()}, args.out)
            print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
