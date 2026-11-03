# KannadaSetu AI

Turn scanned Kannada pages into editable, searchable text — entirely on your own computer.

Upload a PNG, JPG or PDF. Arjuna-OCR reads the page locally, you review the transcription beside the original, correct what the OCR missed, and export or search it later. A "Translate to English" button runs ControlMT locally — no internet, no API keys, no data leaving your machine.

Built with FastAPI, SQLite, and a single-file HTML/JS front end. **100% offline after first setup.**

## Features

- **Local Kannada OCR** via [Arjuna-OCR](https://pypi.org/project/arjuna-ocr/) — ONNX models, CPU-friendly, no GPU required
- **Local Kannada→English translation** via [ControlMT](https://pypi.org/project/controlmt/) (Apache 2.0, ~140 MB int8, CPU inference in well under a second per sentence)
- Upload validation (extension, MIME, file signature, size, page count) with a progress bar
- Per-page OCR with honest per-page status (ok / failed / skipped)
- Side-by-side workspace: original page on the left, editable text and read-only OCR output on the right
- Library in SQLite: full-text search across titles, filenames and Kannada text
- Export as `.txt` or `.json` in UTF-8, with correct `Content-Disposition` for Kannada titles
- Accuracy screen: Character Error Rate (CER) and Word Error Rate (WER) against a reference you paste
- Responsive layout for laptop and phone, keyboard accessible
- **No external APIs** — everything runs on your machine

## Requirements

- Python 3.11 or 3.12 on Windows, macOS or Linux
- ~3 GB free disk: ~2 GB for the virtual environment (torch dominates), ~1 GB for the Hugging Face model cache
- Internet connection on first run only — for `pip install` and for downloading the models
- CPU-only is fine. No GPU needed.

## Quick start (Windows)

1. Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/). Tick **Add python.exe to PATH**.
2. Unzip the project and double-click `run.bat`.
3. The first run creates `.venv`, installs dependencies (a few minutes), and copies `.env.example` to `.env`.
4. Open http://127.0.0.1:8000 in your browser.
5. Upload a Kannada image or PDF. The first OCR run downloads ~240 MB of models and takes 30–60 seconds; later runs are fast.

Other commands: `run.bat setup` reinstalls dependencies, `run.bat test` runs the tests. To use a different port: `set PORT=8010` before `run.bat`.

## Quick start (macOS / Linux / manual)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload