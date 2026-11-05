"""Fine-tune on whole images (no face cropping), adding OpenFake's modern generators.

Run from the project folder:  python src/finetune_full.py
Saves checkpoints/finetuned_full.pt; the other checkpoints are left untouched.
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data import IMAGE_SIZE, MEAN, STD, find_root, get_device
from finetune import FAKE, REAL, evaluate, jpeg_roundtrip, list_images, sample
from model import build_model


def find_dir(root, name):
    return next(p for p in Path(root).rglob("*") if p.is_dir() and p.name == name)


def random_downscale(image, low=320, high=1024):
    """Shrink so the longest side is a random size, so native resolution can't tell real from fake."""
    if random.random() < 0.8:
        longest = max(image.size)
        target = random.randint(low, high)
        if longest > target:
            scale = target / longest
            size = (max(round(image.width * scale), 1), max(round(image.height * scale), 1))
            image = image.resize(size, Image.BILINEAR)
    return image


class WholeImages(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        if train:
            # Random crops with random shape, squashed to a square, so image shape is not a giveaway.
            steps = [
                transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.1),
            ]
        else:
            steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]  # the whole image, as in the app
        self.transform = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label, _group = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.train:
            image = random_downscale(image)
            if random.random() < 0.7:
                image = jpeg_roundtrip(image)
        return self.transform(image), label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/finetuned.pt", help="model to start from")
    parser.add_argument("--out", default="checkpoints/finetuned_full.pt")
    parser.add_argument("--openfake", default="data_extra/openfake")
    parser.add_argument("--new", default="data_extra/diffusion_faces")
    parser.add_argument("--orig-per-class", type=int, default=2000)
    parser.add_argument("--gan-per-class", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra", action="store_true", help="also train on face images from HybridForensics and StyleGAN3")
    parser.add_argument("--hybrid", default="data_extra/hybrid_forensics")
    parser.add_argument("--sg3", default="data_extra/stylegan3_faces")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the code works")
    args = parser.parse_args()

    rng = random.Random(0)
    device = get_device()
    print("Device:", device)

    openfake = next(Path(args.openfake).rglob("openfake_dataset")) / "train"
    new, orig = Path(args.new), find_root("data")

    # Hold 8% of OpenFake's training images back for validation; its test split is never touched here.
    of_train, of_val = {}, {}
    for cls in ("real", "fake"):
        files = list_images(openfake / cls)
        rng.shuffle(files)
        cut = int(len(files) * 0.08)
        of_val[cls], of_train[cls] = files[:cut], files[cut:]

    train_items = (
        [(p, REAL, "real_openfake") for p in of_train["real"]]
        + [(p, FAKE, "fake_openfake") for p in of_train["fake"]]
        + [(p, REAL, "real_new") for p in list_images(new / "train" / "real")]
        + [(p, FAKE, "diffusion") for p in list_images(new / "train" / "diffusion")]
        + [(p, FAKE, "gan_new") for p in sample(list_images(new / "train" / "gan"), args.gan_per_class, rng)]
        + [(p, REAL, "real_orig") for p in sample(list_images(orig / "train" / "real"), args.orig_per_class, rng)]
        + [(p, FAKE, "fake_orig") for p in sample(list_images(orig / "train" / "fake"), args.orig_per_class, rng)]
    )
    val_items = (
        [(p, REAL, "real_openfake") for p in of_val["real"]]
        + [(p, FAKE, "fake_openfake") for p in of_val["fake"]]
        + [(p, REAL, "real_new") for p in list_images(new / "val" / "real")]
        + [(p, FAKE, "diffusion") for p in list_images(new / "val" / "diffusion")]
        + [(p, FAKE, "gan_new") for p in sample(list_images(new / "val" / "gan"), 100, rng)]
        + [(p, REAL, "real_orig") for p in sample(list_images(orig / "valid" / "real"), 300, rng)]
        + [(p, FAKE, "fake_orig") for p in sample(list_images(orig / "valid" / "fake"), 300, rng)]
    )
    if args.extra:
        # Only the face-matched parts of the extra datasets. HybridForensics' SDXL/Midjourney fakes are all food
        # photos, so training on them would teach "food = fake"; they stay unseen, for testing.
        hybrid = find_dir(args.hybrid, "HybridForensics_Dataset_High")
        ffhq = sorted(list_images(hybrid / "Real" / "FFHQ"))
        gan3 = sorted(list_images(hybrid / "Fake_GAN" / "StyleGAN3"))
        progan = sorted(list_images(hybrid / "Fake_GAN" / "ProGAN"))
        sg3_real = list_images(find_dir(args.sg3, "Real faces"))
        sg3_fake = list_images(find_dir(args.sg3, "Fake faces"))
        train_items += (
            [(p, REAL, "real_ffhq") for p in ffhq[:2000]]
            + [(p, FAKE, "fake_gan3") for p in gan3[:1000]]
            + [(p, FAKE, "fake_progan") for p in progan[:1000]]
            + [(p, REAL, "real_sg3") for p in sample(sg3_real, 2000, rng)]
            + [(p, FAKE, "fake_sg3") for p in sample(sg3_fake, 2000, rng)]
        )
        val_items += (
            [(p, REAL, "real_ffhq") for p in ffhq[2000:2150]]
            + [(p, FAKE, "fake_gan3") for p in gan3[1000:1075]]
            + [(p, FAKE, "fake_progan") for p in progan[1000:1075]]
        )
    if args.smoke:
        train_items, val_items = rng.sample(train_items, 192), rng.sample(val_items, 128)
        args.epochs = 1
    print(f"train {len(train_items):,} images, validation {len(val_items):,} images")

    train_loader = DataLoader(WholeImages(train_items, True), args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(WholeImages(val_items, False), 64, num_workers=args.workers)
    val_groups = [item[2] for item in val_items]

    state = torch.load(args.base, map_location=device)
    model = build_model(state["model"], pretrained=False).to(device)
    model.load_state_dict(state["state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    score, per_group = evaluate(model, val_loader, val_groups, device)
    print(f"Before training: balanced acc {score:.3f}  " + "  ".join(f"{g} {v:.2f}" for g, v in per_group.items()))

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
              + "  ".join(f"{g} {v:.2f}" for g, v in per_group.items()), flush=True)
        if score > best and not args.smoke:
            best = score
            Path(args.out).parent.mkdir(exist_ok=True)
            torch.save({"model": state["model"], "state_dict": model.state_dict()}, args.out)
            print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
