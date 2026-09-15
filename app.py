import os
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path='')
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

MODEL_DIR = os.path.join(PROJECT_ROOT, 'Urban_Notebook_results')
ONNX_PATH = os.path.join(MODEL_DIR, 'best_model.onnx')
META_PATH = os.path.join(MODEL_DIR, 'model_meta.json')

# ---------------------------------------------------------------------------
# Model loading — onnxruntime instead of torch/torchvision.
# This removes ~1GB+ of import/runtime memory overhead, which is what was
# causing the free-tier Render instance (512MB RAM) to silently crash mid
# -request during real inference (torch + torchvision + a loaded ResNet-18
# + librosa's own buffers routinely exceeded that ceiling).
# ---------------------------------------------------------------------------
session = None
class_names = None
CONFIG = None

if os.path.exists(ONNX_PATH) and os.path.exists(META_PATH):
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
        class_names = meta["class_names"]
        CONFIG = meta["config"]
        session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        print(f"[OK] ONNX model loaded successfully from {ONNX_PATH}")
        print(f"     Classes: {class_names}")
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model from {ONNX_PATH}: {e}")
        session = None
else:
    print(f"[ERROR] Missing {ONNX_PATH} or {META_PATH}. "
          f"Run the ONNX export step and commit both files.")

LABELS = class_names or [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]

# ---------------------------------------------------------------------------
# Preprocessing — identical to training: Librosa, same sample rate/duration/
# mel params, same per-clip min-max normalization.
# ---------------------------------------------------------------------------
def load_fixed_length_audio(path, sr, duration):
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y

def extract_logmel(y, sr, n_mels, n_fft, hop_length, top_db):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.power_to_db(mel, top_db=top_db)
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-6)
    return logmel.astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_from_file(path):
    """Returns (label, confidence, full_probs_list_in_LABELS_order)."""
    if session is None:
        raise RuntimeError(
            "Model is not loaded — check ONNX_PATH/META_PATH and the server startup logs."
        )

    y = load_fixed_length_audio(path, CONFIG["SAMPLE_RATE"], CONFIG["DURATION"])
    arr = extract_logmel(y, CONFIG["SAMPLE_RATE"], CONFIG["N_MELS"],
                          CONFIG["N_FFT"], CONFIG["HOP_LENGTH"], CONFIG["TOP_DB"])
    x = np.repeat(arr[np.newaxis, :, :], 3, axis=0)[np.newaxis, :, :, :].astype(np.float32)  # (1,3,n_mels,n_frames)

    logits = session.run(None, {"input": x})[0][0]
    probs = softmax(logits)

    top_idx = int(probs.argmax())
    return class_names[top_idx], float(probs[top_idx]), probs.tolist()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        tmp_path = tmp.name
        file.save(tmp_path)

        label, confidence, probabilities = predict_from_file(tmp_path)
        return jsonify({
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "classes": class_names,
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "model_loaded": session is not None,
        "model_path": ONNX_PATH,
        "classes": class_names,
    })

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)