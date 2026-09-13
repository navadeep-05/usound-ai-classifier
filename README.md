# 🎧 Urban Sound Classification AI

> **U‑SOUND AI** – A sleek web interface that lets you upload short city‑environment audio clips and instantly get predictions for 10 common urban sounds using a ResNet‑18 deep‑learning model.

---

## 📁 Project Structure
```
Urban Sound classification/
├─ api/                     # Flask backend (API + static UI)
│   ├─ __pycache__/        # compiled Python files (auto‑generated)
│   ├─ app.py              # main server, model loader, prediction endpoint
│   └─ requirements.txt    # Python dependencies
├─ public/                  # Static assets served by Flask
│   ├─ best_model.pt       # Torch checkpoint (trained model)
│   └─ index.html          # Front‑end UI (styled with glassmorphism)
├─ Test samples/            # Example .wav files for quick testing
├─ Urban_Notebook_results/  # Original notebook artefacts (model, reports)
│   ├─ best_model.pt       # Same model file (mirrored for clarity)
│   ├─ model_scripted.pt   # TorchScript version (optional)
│   └─ *.png / *.txt      # training plots & reports
├─ app.py                    # **Root** Flask entry‑point (static‑file server)
├─ index.html                # Development entry‑point (open directly for UI testing)
├─ requirements.txt          # Top‑level Python deps (mirrors api/requirements.txt)
└─ README.md                 # <-- **This file**
```

*`api/app.py`* hosts the **REST API** (`/predict`, `/health`) and also serves the UI (`/`).
*The UI (`public/index.html`) communicates with the API via a simple `fetch` POST request.*

---

## 🛠️ Core Technologies
- **Python 3.14** – server side.
- **Flask** – lightweight web framework.
- **Flask‑CORS** – enables cross‑origin requests from the UI.
- **torch / torchvision** – model definition & inference.
- **librosa** – audio loading & log‑Mel spectrogram extraction (exactly as used during training).
- **HTML / CSS** – handcrafted UI with a dark‑mode, glass‑morphism look, custom gradients and micro‑animations.

---

## 🚀 Quick Start (Local Development)
1. **Clone / open the repo** – you already have the files.
2. **Create a virtual environment** (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Run the Flask server** (the root `app.py` will start the dev server):
   ```powershell
   cd "c:\Users\navad\Music\Urban Sound classification"
   python app.py
   ```
   You should see something like:
   ```
   * Running on http://127.0.0.1:5000
   ```
5. **Open the UI** – simply double‑click `public/index.html` **or** navigate to `http://127.0.0.1:5000/` in your browser. The page loads the static assets from the Flask server.
6. **Test a prediction** – use the **Upload** button to select any `.wav` file from `Test samples/`. The UI will POST the file to `http://127.0.0.1:5000/predict` and display the top‑5 predictions with confidence bars.

---

## 📡 API End‑points
| Method | Path | Description | Example Curl |
|--------|------|-------------|-------------|
| **POST** | `/predict` | Accepts a multipart/form‑data field `file` (audio). Returns JSON with the predicted label, confidence, full probability vector and class list. | ```bash
curl -X POST -F "file=@Test samples/100795-3-1-0.wav" http://127.0.0.1:5000/predict
``` |
| **GET** | `/health` | Simple health‑check – reports whether the model is loaded and the path it was loaded from. | ```bash
curl http://127.0.0.1:5000/health
``` |
| **GET** | `/` (or any static file) | Serves the UI (`index.html`) and any other assets placed in `public/`. | Open in browser. |

---

## 🧠 How the Model Works (End‑to‑End)
1. **Training (offline)** – a ResNet‑18 backbone was fine‑tuned on the UrbanSound8K dataset. The notebook (`Urban_Notebook_results/`) stores the checkpoint `best_model.pt` containing:
   - `model_state_dict` – learned weights.
   - `class_names` – ordered list of the 10 sound categories.
   - `config` – preprocessing hyper‑parameters (sample rate, mel‑spec settings, etc.).
2. **Loading at runtime** – `app.py` reconstructs the exact architecture via `build_model(num_classes)` and injects the saved state dict. The `CONFIG` dict is also restored for consistent audio preprocessing.
3. **Inference** – when a file is posted:
   - It is saved temporarily on disk (librosa prefers a real file path).
   - `load_fixed_length_audio` → `extract_logmel` generate a **log‑Mel spectrogram** identical to the training pipeline.
   - The spectrogram is reshaped to `(1, 3, n_mels, n_frames)` and fed to the model.
   - Softmax probabilities are computed; the top label and its confidence are returned.
4. **UI Rendering** – the front‑end receives the JSON, maps probabilities to a sleek bar chart, highlights the winning class with a large emoji, and displays a confidence ring.

---

## 🎨 UI Highlights (Why it looks premium)
- **Dark‑mode background** with subtle gradient orbs that float using CSS keyframe animation.
- **Glass‑morphism cards** (`var(--bg-card)`) with backdrop‑filter blur for a polished look.
- **Micro‑animations** on hover (border glow, button scale, icon lift) for an interactive feel.
- **Responsive layout** – the UI flexes from desktop to mobile, keeping the upload zone and result cards accessible.
- **Custom Google Font** – `Inter` for crisp modern typography.
- **Animated spinner** and graceful disabled‑state handling.

---

## 📦 Additional Files & Their Purpose
- **`requirements.txt`** – pin‑pointed Python dependencies (Flask, torch, librosa, etc.).
- **`public/best_model.pt`** – the model used by the API (same as the notebook checkpoint).
- **`Urban_Notebook_results/`** – contains the original training artefacts (training curves, confusion matrix, CSV predictions) – useful for reproducing results or further fine‑tuning.
- **`Test samples/`** – a handful of wav files covering all 10 classes; ideal for quick sanity checks.
- **`index.html` (root)** – a plain static version of the UI, handy for opening directly without the server.

---

## 📖 How to Extend / Contribute
1. **Add new sound classes** – update the training notebook, re‑export the checkpoint, and replace `best_model.pt`.
2. **Swap to TorchScript** – you can use `model_scripted.pt` for faster inference; just replace the loading logic in `app.py` with `torch.jit.load`.
3. **Dockerise** – wrap the Flask server in a lightweight container for production deployment. Example Dockerfile snippet:
   ```dockerfile
   FROM python:3.14-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   ENV PORT=8080
   CMD ["python", "app.py"]
   ```
4. **Front‑end tweaks** – the UI lives entirely in `public/index.html` and its inline CSS; feel free to modernise with a CSS pre‑processor or import a component library.

---

## 🏁 Final Notes
- The server runs in **debug mode** for local development (auto‑reload disabled to avoid path‑related restart issues). For production, switch to a WSGI server like **gunicorn** or **uvicorn**.
- All paths are **relative to the project root**, making the repo portable.
- If you encounter `Model not loaded` errors, verify that `Urban_Notebook_results/best_model.pt` exists and matches the architecture defined in `build_model`.

Enjoy exploring urban sounds! 🎶
