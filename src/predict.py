"""Classify a single photo as real or fake with a trained checkpoint."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from data import IMAGE_SIZE, MEAN, STD, get_device
from model import build_model

# The training images are loose head-and-shoulders crops, so crop wider than the tight face box.
CROP_SCALE = 1.8

FACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_SCORE = 0.8          # ignore detections less confident than this
DETECT_MAX_SIDE = 1024    # shrink big photos before detecting, for speed


class Detector:
    def __init__(self, checkpoint="checkpoints/best.pt"):
        self.device = get_device()
        state = torch.load(checkpoint, map_location=self.device)
        self.model = build_model(state["model"], pretrained=False).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.detector = cv2.FaceDetectorYN.create(
            str(FACE_MODEL), "", (320, 320), score_threshold=FACE_SCORE, nms_threshold=0.3
        )

    def find_face(self, image):
        """Return (x, y, w, h) of the most prominent face in image coordinates, or None."""
        width, height = image.size
        scale = min(1.0, DETECT_MAX_SIDE / max(width, height))
        small = image.resize((max(int(width * scale), 1), max(int(height * scale), 1)))
        bgr = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)

        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.detector.detect(bgr)
        if faces is None:
            return None

        # Prefer big, confident faces. Column 14 is the detection score.
        best = max(faces, key=lambda f: f[2] * f[3] * f[14])
        return [v / scale for v in best[:4]]

    def crop_face(self, image):
        """Return (square crop around the face, True), or (image, False) if no face is found."""
        face = self.find_face(image)
        if face is None:
            return image, False

        x, y, w, h = face
        width, height = image.size
        side = min(int(CROP_SCALE * max(w, h)), width, height)
        left = int(min(max(x + w / 2 - side / 2, 0), width - side))
        top = int(min(max(y + h / 2 - side / 2, 0), height - side))
        return image.crop((left, top, left + side, top + side)), True

    @torch.no_grad()
    def predict(self, image, crop=False):
        """Return a dict with p_real, p_fake, label, the image the model saw, and whether a face was found.

        By default the whole image is analysed. With crop=True it is first cropped around the largest face.
        """
        image = image.convert("RGB")
        face, found = self.crop_face(image) if crop else (image, False)
        batch = self.transform(face).unsqueeze(0).to(self.device)
        p_real = torch.sigmoid(self.model(batch)).item()  # class order is fake=0, real=1
        return {
            "p_real": p_real,
            "p_fake": 1 - p_real,
            "label": "real" if p_real > 0.5 else "fake",
            "face": face,
            "seen": face.resize((IMAGE_SIZE, IMAGE_SIZE)),
            "face_found": found,
        }

    @torch.no_grad()
    def p_fake(self, image):
        """Probability that an already-prepared image is fake."""
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return 1 - torch.sigmoid(self.model(batch)).item()  # class order is fake=0, real=1

    def predict_two_views(self, image, crop_detector=None):
        """Score the whole image and, if a face is found, the face crop too, then average the two.

        The whole image gives context and the face crop gives detail. With no face, only the whole image counts.
        crop_detector: optionally another Detector (model) to score the face crop.
        """
        image = image.convert("RGB")
        p_whole = self.p_fake(image)
        face, found = self.crop_face(image)
        p_crop = (crop_detector or self).p_fake(face) if found else None
        p_fake = p_whole if p_crop is None else (p_whole + p_crop) / 2
        return {
            "p_fake": p_fake,
            "p_real": 1 - p_fake,
            "label": "fake" if p_fake > 0.5 else "real",
            "p_whole": p_whole,
            "p_crop": p_crop,
            "face_found": found,
            "whole_seen": image.resize((IMAGE_SIZE, IMAGE_SIZE)),
            "face_seen": face.resize((IMAGE_SIZE, IMAGE_SIZE)) if found else None,
        }
