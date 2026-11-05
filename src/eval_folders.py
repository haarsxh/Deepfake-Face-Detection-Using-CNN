"""Test a checkpoint on image folders it was not trained on, with the same face cropping the app uses.

Example:
  python src/eval_folders.py --root data_extra/diffusion_faces --split test \
      --real real --fake gan diffusion photoshop
"""
import argparse
import io
import random
from pathlib import Path

from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from predict import Detector

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(root, name, split):
    """All images under folders called `name` (inside the `split` folder, if one is given)."""
    images = []
    for folder in Path(root).rglob("*"):
        if folder.is_dir() and folder.name.lower() == name.lower() and (split is None or split in folder.parts):
            images += [p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS]
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default=None, help="only use folders inside this one, e.g. test")
    parser.add_argument("--real", nargs="+", required=True, help="folder names that hold real photos")
    parser.add_argument("--fake", nargs="+", required=True, help="folder names that hold fake photos")
    parser.add_argument("--limit", type=int, default=None, help="max images per folder")
    parser.add_argument("--crop", action="store_true", help="crop to the face first (default: whole image)")
    parser.add_argument("--normalize", action="store_true",
                        help="shrink every image to 512 px on its longest side and recompress as JPEG, "
                             "so size and format can't give the answer away")
    args = parser.parse_args()

    detector = Detector(args.checkpoint)
    random.seed(0)

    rows = []  # (folder name, is_fake, p_fake, face_found)
    for names, is_fake in ((args.real, 0), (args.fake, 1)):
        for name in names:
            images = find_images(args.root, name, args.split)
            if args.limit and len(images) > args.limit:
                images = random.sample(images, args.limit)
            if not images:
                print(f"No images found for folder '{name}'")
                continue
            for path in tqdm(images, desc=name, leave=False):
                image = Image.open(path).convert("RGB")
                if args.normalize:
                    scale = 512 / max(image.size)
                    image = image.resize((max(round(image.width * scale), 1), max(round(image.height * scale), 1)))
                    buffer = io.BytesIO()
                    image.save(buffer, "JPEG", quality=90)
                    image = Image.open(buffer).convert("RGB")
                result = detector.predict(image, crop=args.crop)
                rows.append((name, is_fake, result["p_fake"], result["face_found"]))

    print(f"\n{'folder':<12}{'images':>8}{'said fake':>12}{'said real':>11}{'face found':>12}")
    for name in args.real + args.fake:
        group = [r for r in rows if r[0] == name]
        if not group:
            continue
        fake_rate = sum(r[2] > 0.5 for r in group) / len(group)
        found = sum(r[3] for r in group) / len(group)
        kind = "(real)" if name in args.real else "(fake)"
        print(f"{name + ' ' + kind:<12}{len(group):>8}{fake_rate:>12.1%}{1 - fake_rate:>11.1%}{found:>12.1%}")

    real = [r for r in rows if not r[1]]
    fake = [r for r in rows if r[1]]
    if real and fake:
        real_recall = sum(r[2] <= 0.5 for r in real) / len(real)
        fake_recall = sum(r[2] > 0.5 for r in fake) / len(fake)
        auc = roc_auc_score([r[1] for r in rows], [r[2] for r in rows])
        print(f"\nreal photos correct: {real_recall:.1%}   fakes caught: {fake_recall:.1%}")
        print(f"balanced accuracy: {(real_recall + fake_recall) / 2:.1%}   ROC AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
