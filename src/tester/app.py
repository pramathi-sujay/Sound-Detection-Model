"""
AIISH Model Tester
------------------
Internal Flask utility for evaluating AIISH_v2.tflite on audio recordings.

Pipeline:
    Audio File → load_audio() → YAMNet → Mean Embedding (1024)
    → AIISH_v2.tflite → Prediction

Run:
    python src/tester/app.py

Then open:
    http://127.0.0.1:5000

Author: Pramathi Sujay
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Ensure repository root is on the path when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preproc import load_audio
from src.model.yamnet_loader import load_yamnet


# ==========================================================
# Paths / constants
# ==========================================================

TFLITE_MODEL = REPO_ROOT / "models" / "AIISH_v2.tflite"
LABEL_PATH = REPO_ROOT / "data" / "mappings" / "unified_labels.json"
MODEL_DISPLAY_NAME = TFLITE_MODEL.name

TOP_K = 5
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

TESTER_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(TESTER_DIR / "templates"),
    static_folder=str(TESTER_DIR / "static"),
)
app.secret_key = "aiish-internal-tester"


# Loaded once at startup
yamnet_model = None
tflite_interpreter = None
class_names: list[str] = []


# ==========================================================
# Model / label loading
# ==========================================================

def load_labels(label_path: Path) -> list[str]:
    """Load index → class name mapping used by AIISH_v2."""
    with open(label_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return [label_map[str(i)] for i in range(len(label_map))]


def load_tflite_interpreter(model_path: Path):
    """Create and allocate a TFLite interpreter for AIISH_v2."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def initialize_models() -> None:
    """Load YAMNet, TFLite classifier, and labels once at startup."""
    global yamnet_model, tflite_interpreter, class_names

    if not TFLITE_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {TFLITE_MODEL}")
    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Missing labels: {LABEL_PATH}")

    class_names = load_labels(LABEL_PATH)
    yamnet_model = load_yamnet()
    tflite_interpreter = load_tflite_interpreter(TFLITE_MODEL)


# ==========================================================
# Inference
# ==========================================================

def predict_audio(audio_path: str) -> dict:
    """
    Run the full AIISH inference pipeline on one audio file.

    Reuses:
        - load_audio() for mono 16 kHz preprocessing
        - YAMNet for 1024-d frame embeddings
        - mean pooling (same as training)
        - AIISH_v2.tflite for classification
    """
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    start = time.perf_counter()

    # 1. Load audio as mono 16 kHz float32
    waveform = load_audio(audio_path)
    duration_s = float(len(waveform) / 16000.0)

    # 2. YAMNet embedding (num_frames, 1024)
    _, embedding, _ = yamnet_model(waveform)

    # 3. Mean-pool to a single 1024-d vector
    embedding = tf.reduce_mean(embedding, axis=0).numpy()
    embedding_shape = tuple(int(dim) for dim in embedding.shape)
    sample = embedding.astype(np.float32).reshape(1, -1)

    # 4. AIISH TFLite classification
    tflite_interpreter.set_tensor(input_details[0]["index"], sample)
    tflite_interpreter.invoke()
    probabilities = tflite_interpreter.get_tensor(output_details[0]["index"])[0]

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # 5. Rank predictions
    ranked_indices = np.argsort(probabilities)[::-1]
    top_indices = ranked_indices[:TOP_K]
    predicted_idx = int(ranked_indices[0])
    confidence = float(probabilities[predicted_idx])

    top_predictions = [
        {
            "rank": rank,
            "label": class_names[int(idx)],
            "probability": float(probabilities[int(idx)]),
            "percent": float(probabilities[int(idx)]) * 100.0,
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

    return {
        "filename": Path(audio_path).name,
        "predicted_class": class_names[predicted_idx],
        "confidence": confidence,
        "confidence_percent": confidence * 100.0,
        "top_predictions": top_predictions,
        "inference_ms": elapsed_ms,
        "duration_s": duration_s,
        "embedding_shape": embedding_shape,
        "model_name": MODEL_DISPLAY_NAME,
    }


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ==========================================================
# Routes
# ==========================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("audio_file")

    if uploaded is None or uploaded.filename is None or uploaded.filename.strip() == "":
        flash("No audio file selected. Please choose a file and try again.")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded.filename)
    if not allowed_file(filename):
        flash(
            "Unsupported audio format. "
            "Please upload wav, mp3, flac, ogg, m4a, or aac."
        )
        return redirect(url_for("index"))

    suffix = Path(filename).suffix.lower()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)

        result = predict_audio(str(temp_path))
        result["filename"] = filename
        return render_template("index.html", result=result)

    except Exception as exc:
        flash(f"Inference failed: {exc}")
        return redirect(url_for("index"))

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ==========================================================
# Entry point
# ==========================================================

def main():
    print("=" * 60)
    print("AIISH Model Tester")
    print("=" * 60)

    try:
        initialize_models()
    except Exception as exc:
        print(f"\nStartup failed: {exc}")
        print("Fix the missing asset / dependency and try again.")
        sys.exit(1)

    print(f"\nModel : {MODEL_DISPLAY_NAME}")
    print(f"Classes: {len(class_names)}")
    print("\nOpen in browser: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
