"""
detect.py — YOLO Face Detection + EmotionCNN Pipeline
Class names match the exact folder structure of the user's dataset:
  data/
    train/  angry/ disgusted/ fearful/ happy/ neutral/ sad/ surprised/
    test/   angry/ disgusted/ fearful/ happy/ neutral/ sad/ surprised/
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import os

# ── Emotion labels — MUST match folder names exactly ──────────────────────────
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Display-friendly capitalised versions
EMOTION_DISPLAY = {
    "angry":     "Angry",
    "disgusted": "Disgusted",
    "fearful":   "Fearful",
    "happy":     "Happy",
    "neutral":   "Neutral",
    "sad":       "Sad",
    "surprised": "Surprised",
}

EMOTION_COLORS = {          # BGR for OpenCV drawing
    "angry":     (0,   30, 255),
    "disgusted": (0,  140,   0),
    "fearful":   (180,  0, 180),
    "happy":     (0,  215, 255),
    "neutral":   (180,180, 180),
    "sad":       (255, 80,   0),
    "surprised": (0,  200, 100),
}

EMOTION_HEX = {             # Hex for CSS / Streamlit
    "angry":     "#ef4444",
    "disgusted": "#22c55e",
    "fearful":   "#a855f7",
    "happy":     "#eab308",
    "neutral":   "#94a3b8",
    "sad":       "#3b82f6",
    "surprised": "#f97316",
}

EMOTION_EMOJIS = {
    "angry":     "😠",
    "disgusted": "🤢",
    "fearful":   "😨",
    "happy":     "😊",
    "neutral":   "😐",
    "sad":       "😢",
    "surprised": "😲",
}


class EmotionCNN(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# ── Pre-processing transform for inference ─────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ── Main detector class ────────────────────────────────────────────────────────
class EmotionDetector:
    """
    Two-stage pipeline:
      Stage 1 — YOLOv8 detects face bounding boxes in the image.
      Stage 2 — EmotionCNN classifies each cropped face into one of 7 emotions.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Redirect YOLO downloads to /tmp so Streamlit Cloud (read-only home) works
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

        # ── Stage 1: YOLO face detector ───────────────────────────────────────
        # BUG #2 FIX: Never fall back to yolov8n.pt (general object detector).
        # It detects whole persons (torso + background), NOT faces.
        # Sending a full-body crop to EmotionCNN causes wrong predictions.
        # Correct fallback order:
        #   1. yolov8n-face.pt (local file, committed to repo)
        #   2. Download from HuggingFace Hub
        #   3. OpenCV Haar Cascade (actually detects faces, unlike yolov8n.pt)

        self.yolo = None
        self.face_cascade = None

        # Priority 1: local face model file
        if os.path.exists("yolov8n-face.pt"):
            self.yolo = YOLO("yolov8n-face.pt")
            print("[✓] Loaded yolov8n-face.pt from local file.")

        # Priority 2: download face model from HuggingFace Hub
        if self.yolo is None:
            try:
                from huggingface_hub import hf_hub_download
                face_model_path = hf_hub_download(
                    repo_id="arnabdhar/YOLOv8-Face-Detection",
                    filename="model.pt",
                    cache_dir="/tmp/yolo_face",
                )
                self.yolo = YOLO(face_model_path)
                print("[✓] Loaded YOLOv8 face model from HuggingFace Hub.")
            except Exception as e:
                print(f"[!] HuggingFace download failed: {e}")

        # Priority 3: OpenCV Haar Cascade — detects FACES (correct fallback)
        # NOTE: yolov8n.pt is intentionally NOT used here — it is a general
        # 80-class COCO detector that crops whole persons, not faces.
        if self.yolo is None:
            print("[!] Falling back to OpenCV Haar Cascade face detector.")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.face_cascade.empty():
                raise RuntimeError(
                    "Haar Cascade XML not found. "
                    "Ensure opencv-python or opencv-python-headless is installed."
                )
            print("[✓] OpenCV Haar Cascade loaded as face detector fallback.")

        # ── Stage 2: Emotion CNN ──────────────────────────────────────────────
        self.emotion_model = EmotionCNN(num_classes=len(EMOTIONS)).to(self.device)

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            self.emotion_model.load_state_dict(state)
            print(f"[✓] Emotion model loaded from: {model_path}")
        else:
            print("[!] No trained weights found — predictions will be random.")
            print("    Run:  python train.py  to train the model first.")

        self.emotion_model.eval()

    # ── Detect all faces in a BGR frame ───────────────────────────────────────
    def detect_faces(self, frame_bgr: np.ndarray) -> list:
        """Returns [(x1,y1,x2,y2,conf), ...] for every face found."""

        # ── YOLO face model path ───────────────────────────────────────────────
        if self.yolo is not None:
            results = self.yolo(frame_bgr, verbose=False)[0]
            boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))
            return boxes

        # ── Haar Cascade fallback path ─────────────────────────────────────────
        # Haar has no confidence score; use 0.99 as a placeholder so the rest
        # of the pipeline (which expects a 5-tuple) works unchanged.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),   # skip detections too small for the CNN
        )
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append((x, y, x + w, y + h, 0.99))
        return boxes

    # ── Classify emotion from a BGR face crop ─────────────────────────────────
    def classify_emotion(self, face_bgr: np.ndarray) -> tuple:
        """Returns (label, confidence, {label: prob, ...})."""
        pil_face = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor   = INFER_TRANSFORM(pil_face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.emotion_model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        idx   = int(np.argmax(probs))
        label = EMOTIONS[idx]
        conf  = float(probs[idx])
        prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        return label, conf, prob_dict

    # ── Full pipeline: any image input → annotated BGR + detection list ────────
    def process_image(self, image_input, draw_boxes: bool = True) -> tuple:
        """
        Accepts:  np.ndarray (BGR) | PIL.Image | str (file path)
        Returns:  (annotated_bgr: np.ndarray,  detections: list[dict])

        Each detection dict contains:
          bbox, face_confidence, emotion, display_emotion,
          confidence, probabilities, emoji, color_hex
        """
        # Normalise to BGR numpy array
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            frame = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            frame = image_input.copy()

        h, w = frame.shape[:2]
        detections = []

        for (x1, y1, x2, y2, face_conf) in self.detect_faces(frame):
            # Clamp crop to image bounds
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                continue

            pad_x = int((cx2 - cx1) * 0.15)
            pad_y = int((cy2 - cy1) * 0.15)

            nx1 = max(0, cx1 - pad_x)
            ny1 = max(0, cy1 - pad_y)

            nx2 = min(w, cx2 + pad_x)
            ny2 = min(h, cy2 + pad_y)

            crop = frame[ny1:ny2, nx1:nx2]
            label, emo_conf, probs = self.classify_emotion(crop)
            bgr_color = EMOTION_COLORS[label]

            detections.append({
                "bbox":             (x1, y1, x2, y2),
                "face_confidence":  face_conf,
                "emotion":          label,                      # raw key, e.g. "angry"
                "display_emotion":  EMOTION_DISPLAY[label],     # capitalised
                "confidence":       emo_conf,
                "probabilities":    probs,
                "emoji":            EMOTION_EMOJIS[label],
                "color_hex":        EMOTION_HEX[label],
            })

            if draw_boxes:
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

                # Label banner above the box
                label_text = f"{EMOTION_EMOJIS[label]} {EMOTION_DISPLAY[label]} {emo_conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                banner_y1 = max(0, y1 - th - 10)
                cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 8, y1), bgr_color, -1)
                cv2.putText(
                    frame, label_text, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

        return frame, detections