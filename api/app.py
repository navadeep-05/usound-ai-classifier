# api/app.py
import os
import json
from flask import Flask, request, jsonify
import torch
import torchaudio
import io

app = Flask(__name__)

# Load the model once – Vercel keeps the function warm between invocations
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'public', 'best_model.pt')
device = torch.device('cpu')
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

def predict(audio_bytes):
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    # TODO: add any preprocessing you need here
    with torch.no_grad():
        out = model(waveform.to(device))
    probs = torch.softmax(out, dim=1).cpu().numpy().tolist()[0]
    return probs

@app.route('/api/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    file = request.files['file']
    probs = predict(file.read())
    return jsonify({'probs': probs})

def handler(request, context):
    """Vercel server‑less entry point."""
    with app.test_request_context(
        path=request.path,
        method=request.method,
        query_string=request.query_string,
        headers=request.headers,
        data=request.body,
        json=request.json,
    ):
        resp = app.full_dispatch_request()
        return {
            'statusCode': resp.status_code,
            'headers': dict(resp.headers),
            'body': resp.get_data(as_text=True),
        }

import tempfile
import numpy as np
from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import librosa

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the directory containing app.py and all project files.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path='')
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable caching for static assets

# The real checkpoint from the Kaggle notebook, unzipped into this folder.
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Urban_Notebook_results', 'best_model.pt')

# ---------------------------------------------------------------------------
# Model loading — rebuild the ResNet18 architecture, then load the weights.
# The checkpoint is a dict (model_state_dict + class_names + config), not a
# raw model object, so torch.load(...).eval() (the old code) can never work.
# ---------------------------------------------------------------------------
def build_model(num_classes):
    m = torchvision.models.resnet18(weights=None)
    in_feats = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feats, num_classes))
    return m

model = None
class_names = None
CONFIG = None

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        CONFIG = checkpoint["config"]
        class_names = checkpoint["class_names"]          # always read from checkpoint, never hardcode
        model = build_model(len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[OK] Model loaded successfully from {MODEL_PATH}")
        print(f"     Classes: {class_names}")
        print(f"     Training val acc: {checkpoint.get('best_val_acc', 'n/a')}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
        model = None
else:
    print(f"[ERROR] Model file not found at {MODEL_PATH}. "
          f"Check that Urban_Notebook_results/best_model.pt exists.")

# Fallback label list (only used if the checkpoint truly can't be loaded,
# so the API doesn't crash — but this should never be hit once the path
# above is correct).
LABELS = class_names or [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]

# ---------------------------------------------------------------------------
# Preprocessing — MUST exactly match what the notebook used for training:
# librosa (not torchaudio), same sample rate / duration / mel params, and
# the same per-clip min-max normalization.
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

def predict_from_file(path, top_k=5):
    """Returns (label, confidence, full_probs_list_in_LABELS_order)."""
    if model is None:
        raise RuntimeError(
            "Model is not loaded — check the MODEL_PATH and the server startup logs."
        )

    y = load_fixed_length_audio(path, CONFIG["SAMPLE_RATE"], CONFIG["DURATION"])
    arr = extract_logmel(y, CONFIG["SAMPLE_RATE"], CONFIG["N_MELS"],
                          CONFIG["N_FFT"], CONFIG["HOP_LENGTH"], CONFIG["TOP_DB"])
    x = torch.from_numpy(arr).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # (1,3,n_mels,n_frames)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].numpy()

    top_idx = int(probs.argmax())
# (Removed legacy Flask routes – Vercel will use the /api/classify endpoint defined earlier)

if __name__ == '__main__':
    # Local development convenience – run the API on port 5000
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)