import io
import json
import logging
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
from torchvision import models

from config import CLASSES, IMG_SIZE


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "best_model.pt"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FILE_FIELD = "image"
MODEL_CLASSES = sorted(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/script.js": ("script.js", "text/javascript; charset=utf-8"),
}
EVAL_TRANSFORM = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()


def send_json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        asset = ALLOWED_FILES.get(path)
        if asset is None:
            send_json(self, 404, {"error": "Page not found."})
            return

        filename, content_type = asset
        body = (ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlsplit(self.path).path != "/predict":
            send_json(self, 404, {"error": "Prediction endpoint not found."})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            send_json(self, 415, {"error": "Upload an image using multipart/form-data."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            send_json(self, 411, {"error": "A valid Content-Length header is required."})
            return

        if content_length < 1:
            send_json(self, 400, {"error": "The upload is empty."})
            return
        if content_length > MAX_UPLOAD_BYTES:
            send_json(self, 413, {"error": "Image must be smaller than 20 MB."})
            return

        try:
            body = self.rfile.read(content_length)
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
                + body
            )
            image_part = next(
                (
                    part
                    for part in message.iter_parts()
                    if part.get_content_disposition() == "form-data"
                    and part.get_param("name", header="content-disposition") == FILE_FIELD
                ),
                None,
            )
            if image_part is None or image_part.get_payload(decode=True) is None:
                send_json(self, 400, {"error": "Choose an image before running a prediction."})
                return

            image_bytes = image_part.get_payload(decode=True)
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")
                tensor = EVAL_TRANSFORM(image=np.asarray(rgb_image))["image"].unsqueeze(0).to(DEVICE)
        except (UnidentifiedImageError, OSError, ValueError):
            send_json(self, 400, {"error": "The uploaded file is not a valid image."})
            return
        except Exception:
            logging.exception("Failed to read uploaded image")
            send_json(self, 400, {"error": "The uploaded image could not be processed."})
            return

        try:
            with torch.inference_mode():
                probabilities = torch.softmax(MODEL(tensor), dim=1)[0]
            confidence, index = probabilities.max(dim=0)
            send_json(
                self,
                200,
                {"class": MODEL_CLASSES[index.item()], "confidence": confidence.item()},
            )
        except Exception:
            logging.exception("Model inference failed")
            send_json(self, 500, {"error": "The model could not complete this prediction."})

    def log_message(self, format_string, *args):
        logging.info("%s - %s", self.address_string(), format_string % args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print(f"Image classifier running at http://127.0.0.1:8000 (device: {DEVICE})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping image classifier.")
    finally:
        server.server_close()
