"""Train a real/fake face classifier."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_device, make_loader
from model import build_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.float().to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += ((logits > 0).float() == labels).sum().item()
            count += len(labels)

    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", choices=["simple", "efficientnet"], default="efficientnet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-subset", type=int, default=None, help="use only N training images")
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    train_loader, _ = make_loader(args.data, "train", args.batch_size, args.workers, args.train_subset)
    val_loader, _ = make_loader(args.data, "valid", args.batch_size, args.workers)

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.4f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": args.model, "state_dict": model.state_dict()}, out_dir / "best.pt")
            print(f"  saved best model (val acc {best_acc:.4f})")


if __name__ == "__main__":
    main()
