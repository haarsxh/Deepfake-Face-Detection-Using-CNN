"""Compare whole-image, face-crop and combined (two-view) predictions on folders of real and fake images.

Each image is scored once per view and the rules are compared offline, so every rule sees exactly the same images.

Example:
  python src/eval_two_view.py --root data_extra/diffusion_faces --split test --real real --fake gan diffusion
"""
import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from eval_folders import find_images
from predict import Detector


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def rules(row):
    """The prediction rules being compared: each maps one image's scores to a probability of 'fake'."""
    whole = row["p_whole"]
    crop = row["p_crop"] if row["face_found"] else whole
    crop_ft = row["p_crop_ft"] if row["face_found"] else whole
    return {
        "whole image only": whole,
        "face crop only (whole if no face)": crop,
        "two views: average": (whole + crop) / 2,
        "two views: max (either says fake)": max(whole, crop),
        "two views: whole + crop-trained model's crop": (whole + crop_ft) / 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/finetuned_final.pt")
    parser.add_argument("--crop-checkpoint", default="checkpoints/finetuned.pt",
                        help="a model trained on face crops, tried as the face-crop scorer")
    parser.add_argument("--split", default=None)
    parser.add_argument("--real", nargs="+", required=True)
    parser.add_argument("--fake", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None, help="max images per folder")
    parser.add_argument("--save", default=None, help="write per-image scores to this CSV")
    args = parser.parse_args()

    main_model = Detector(args.checkpoint)
    crop_model = Detector(args.crop_checkpoint)
    random.seed(0)

    rows = []
    for names, is_fake in ((args.real, 0), (args.fake, 1)):
        for name in names:
            images = find_images(args.root, name, args.split)
            if args.limit and len(images) > args.limit:
                images = random.sample(images, args.limit)
            for path in tqdm(images, desc=name, leave=False):
                image = Image.open(path).convert("RGB")
                face, found = main_model.crop_face(image)
                rows.append({
                    "folder": name,
                    "is_fake": is_fake,
                    "face_found": found,
                    "p_whole": main_model.p_fake(image),
                    "p_crop": main_model.p_fake(face) if found else None,
                    "p_crop_ft": crop_model.p_fake(face) if found else None,
                })

    if args.save:
        Path(args.save).parent.mkdir(exist_ok=True)
        with open(args.save, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    found = sum(r["face_found"] for r in rows)
    print(f"{len(rows)} images, a face was found in {found} ({found / len(rows):.0%})\n")
    print(f"{'rule':<46}{'real ok':>9}{'fakes caught':>14}{'balanced':>10}{'AUC':>7}")
    labels = [r["is_fake"] for r in rows]
    for name in rules(rows[0]):
        scores = [rules(r)[name] for r in rows]
        real = [s <= 0.5 for s, y in zip(scores, labels) if y == 0]
        fake = [s > 0.5 for s, y in zip(scores, labels) if y == 1]
        real_ok, fake_ok = sum(real) / len(real), sum(fake) / len(fake)
        print(f"{name:<46}{real_ok:>9.1%}{fake_ok:>14.1%}{(real_ok + fake_ok) / 2:>10.1%}"
              f"{roc_auc_score(labels, scores):>7.3f}")


if __name__ == "__main__":
    main()
