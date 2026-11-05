"""Download and extract the 140k Real and Fake Faces dataset from Kaggle.

Needs Kaggle credentials (see README). The dataset is about 4 GB.
"""
import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "xhlulu/140k-real-and-fake-faces"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="folder to extract into")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Prefer the kaggle CLI from the same environment as this Python, so activating the venv isn't required.
    venv_kaggle = Path(sys.executable).parent / "kaggle"
    kaggle = str(venv_kaggle) if venv_kaggle.exists() else "kaggle"
    cmd = [kaggle, "datasets", "download", "-d", DATASET, "-p", str(out), "--unzip"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("The 'kaggle' command was not found. Run: pip install -r requirements.txt")
    except subprocess.CalledProcessError:
        sys.exit("Download failed. Check your Kaggle credentials (see README).")

    print(f"Done. Data is in {out.resolve()}")


if __name__ == "__main__":
    main()
