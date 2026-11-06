# Deepfake Face Detection Using CNN

A real-vs-AI-generated image classifier (EfficientNet-B0, PyTorch) with a small web app: upload a photo and
see whether the model thinks it is real or fake. It analyses the **whole image**; face cropping is optional.

The model was trained in stages, each added to fix a weakness found by testing on images it had not seen:

1. **140k Real and Fake Faces** (StyleGAN faces), trained on a 5,000-image subset for 2 epochs: 94.7% test accuracy
   on that dataset, but it caught only 1% of diffusion fakes.
2. **+ Real/Photoshop/GAN/Diffusion faces**: diffusion fakes caught 1% to 99%.
3. **+ OpenFake** (whole images from recent generators): OpenFake fakes caught 28% to 91%.
4. **+ HybridForensics / StyleGAN3 faces** (face parts only): more GAN generators covered.

## Results (images never used in training)

| Test | Final model |
|---|---|
| OpenFake test, balanced accuracy | 88.2% (ROC AUC 0.958) |
| Same test, all images shrunk to equal size | 88.5% |
| SDXL + Midjourney fakes vs real COCO scenes (unseen generators) | 78.4% (fakes caught 98.5%, real scenes correct 58%) |
| Original 140k face test, balanced accuracy | 95.5% |
| New diffusion-faces test, balanced accuracy | 86.4% (Photoshop-edited fakes are not detected, by design) |

Raw outputs are in `results/`.

### Limits to know about

- **False alarms:** about 13% of real photos are called fake on OpenFake's test, about 42% of everyday COCO scene
  photos, and 8% of real faces. A "fake" verdict on an ordinary photo is a hint, not proof.
- **Unseen generators:** the score on a generator the model has not seen can be far below the numbers above.
- **Dataset shortcuts:** OpenFake's real images are everyday web photos and its fakes are polished AI images, so some of
  what the model learned may be style, not generator artifacts. HybridForensics' SDXL/Midjourney fakes are all food
  photos, so they were used only for testing, never for training.
- Not covered: face-swap videos, Photoshop edits, newer generators than the data contains.

### Two-view check (tested, not enabled in the app)

`Detector.predict_two_views` scores the whole image and the face crop and averages them; `src/eval_two_view.py`
compares it with the single views on the same images (`results/two_view_tests.txt`). It did not clearly help:

| Test (balanced accuracy) | Whole image | Face crop only | Average of both |
|---|---|---|---|
| New diffusion faces / original 140k faces | 95.6% / 95.5% | same | same (face-only images are already "cropped") |
| OpenFake test (a face found in 75%) | **88.2%** | 75.7% | 86.1% |
| SDXL + Midjourney vs COCO (a face found in 9%) | 78.4% | 81.9% | 81.6% |

Cropping lowers false alarms on real photos but also misses many fakes, because this model uses image context as well
as face detail. A two-view model would need to be trained on crops of the OpenFake people images too.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.12 (or 3.11); PyTorch support for 3.14 is not guaranteed. The pretrained model is in
`checkpoints/finetuned_final.pt`.

## Web app

```bash
python src/app.py
```

Open http://127.0.0.1:7860, upload a photo and press Check photo. The whole image is analysed (resized to 224x224).
Tick "Crop to the face first" to crop around the largest face (YuNet detector in `models/`). Pass
`--checkpoint <file>` to use a different model.

## Reproduce the training

Kaggle credentials are needed (create a token on kaggle.com, Settings, API; save it as `~/.kaggle/access_token`
or run `kaggle auth login`). Datasets go in `data/` and `data_extra/` (not stored in this repository).

```bash
python src/download_data.py                                   # 140k Real and Fake Faces, into data/
kaggle datasets download -d fatimasalman/real-photoshop-gan-diffusion-faces -p data_extra/diffusion_faces --unzip
kaggle datasets download -d sanketghadge1/openfake-data-20k-img -p data_extra/openfake --unzip
kaggle datasets download -d shanmuk4622/real-and-fake-ai-generated-512px-dataset -p data_extra/hybrid_forensics --unzip
kaggle datasets download -d troykueh/real-vs-fake-faces-stylegan3 -p data_extra/stylegan3_faces --unzip

python src/train.py --train-subset 5000 --epochs 2            # stage 1: base model (as used here; drop the subset for a full run)
python src/finetune.py                                        # stage 2: diffusion faces (face crops)
python src/finetune_full.py                                   # stage 3: OpenFake mix, whole images
python src/finetune_full.py --extra --base checkpoints/finetuned_full.pt --out checkpoints/finetuned_final.pt --epochs 3 --lr 2e-5
```

Test on folders of images with `src/eval_folders.py`, for example:

```bash
python src/eval_folders.py --checkpoint checkpoints/finetuned_final.pt --root data_extra/openfake --split test \
    --real real --fake fake --limit 600
```

Add `--normalize` to shrink every image to the same size first, and `--crop` to crop faces.

## Files

- `src/data.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`: the original face-only training and evaluation
- `src/finetune.py`: stage 2 (face crops, diffusion faces)
- `src/finetune_full.py`: stages 3 and 4 (whole images, with JPEG recompression, random downscaling and
  random-shape crops during training so file format, resolution and image shape are not shortcuts)
- `src/predict.py`: loading a model and classifying one image; `src/app.py`: the web app
- `src/eval_folders.py`: test any folders of real and fake images
- `notebooks/kaggle_train.ipynb`: the original training as a Kaggle notebook with a live progress panel.
  Kaggle only gives a GPU and Internet to phone-verified accounts; without them it trains on the CPU and from scratch.
- `results/`: saved test and training outputs

Labels follow folder order: `fake` = 0, `real` = 1.

## License

MIT, see [LICENSE](LICENSE). The datasets used for training have their own licences and are not included.
