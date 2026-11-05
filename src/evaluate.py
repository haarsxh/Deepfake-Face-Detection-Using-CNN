"""Evaluate a trained checkpoint on the test split."""
import argparse

import matplotlib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from data import get_device, make_loader
from model import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader, class_names = make_loader(args.data, args.split, args.batch_size, args.workers)

    scores, labels = [], []
    with torch.no_grad():
        for images, y in tqdm(loader):
            logits = model(images.to(device)).squeeze(1)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    preds = (scores > 0.5).astype(int)

    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print("ROC AUC:", round(roc_auc_score(labels, scores), 4))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
