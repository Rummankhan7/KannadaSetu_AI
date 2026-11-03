"""KannadaSetu AI - Kannada document OCR workspace.
Single-file FastAPI backend using arjuna-ocr for local inference.
Run:  uvicorn app:app --host 127.0.0.1 --port 8000
"""
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ocr_engine import KannadaOCREngine
from correction_engine import CorrectionEngine
from export_engine import ExportEngine

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VERSION = "3.0.0"
log = logging.getLogger("kannadasetu")

MAX_EVAL_CHARS = 20_000
OCR_MAX_SIDE = 2200
PREVIEW_MAX_SIDE = 1800
Image.MAX_IMAGE_PIXELS = 80_000_000


def _env_number(name: str, default, cast):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass
class Settings:
    max_upload_mb: int = 15
    max_pdf_pages: int = 10
    data_dir: Path = BASE_DIR / "data"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")
        data_dir = Path(os.getenv("KANNADASETU_DATA_DIR", "").strip() or "data")
        if not data_dir.is_absolute():
            data_dir = BASE_DIR / data_dir
        return cls(
            max_upload_mb=_env_number("MAX_UPLOAD_MB", 15, int),
            max_pdf_pages=_env_number("MAX_PDF_PAGES", 10, int),
            data_dir=data_dir,
        )

    @property
    def upload_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "kannadasetu.db"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024


# ---------------------------------------------------------------------------
# 2. Errors
# ---------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra


class ProviderError(Exception):
    def __init__(self, code: str, message: str, status: int = 502, retry_after: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retry_after = retry_after


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 3. OCR provider — arjuna-ocr + ControlMT (fully local)
# ---------------------------------------------------------------------------
class ArjunaLocalProvider:
    name = "arjuna-ocr-local"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_success_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self._engine: Optional[KannadaOCREngine] = None
        self._mt = None
        self._load_lock = threading.Lock()

    @property
    def config_problem(self) -> Optional[str]:
        try:
            import kanen_infer  # noqa
            return None
        except ImportError:
            return "arjuna-ocr is not installed. Run: pip install arjuna-ocr"

    @property
    def configured(self) -> bool:
        return self.config_problem is None

    def _load_engine(self) -> KannadaOCREngine:
        if self._engine is not None:
            return self._engine
        with self._load_lock:
            if self._engine is not None:
                return self._engine
            problem = self.config_problem
            if problem:
                raise ProviderError("ocr_not_configured", problem, 503)
            try:
                log.info("Loading Arjuna-OCR engine...")
                self._engine = KannadaOCREngine()
                self._engine.warmup()
                log.info("Arjuna-OCR engine ready")
            except Exception as e:
                log.error("Failed to load arjuna-ocr: %s", e)
                raise ProviderError(
                    "ocr_model_load_failed",
                    f"Could not load the OCR model: {e}. Check your internet connection "
                    "(first run downloads ~170 MB) or ARJUNA_MODELS path.",
                    500,
                )
        return self._engine

    def recognize(self, image_bytes: bytes, mime: str, filename: str,
                  mode: str = "auto", ensemble: bool = True) -> dict:
        engine = self._load_engine()
        import tempfile
        import os as _os
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            with Image.open(tmp_path) as img:
                result = engine.recognize(img, mode=mode, enhance=True, ensemble=ensemble)
            self.last_success_at = now_iso()
            self.last_error = None
            return {
                "text": clean_ocr_text(result.text),
                "confidence": float(result.confidence),
                "words": result.words[:5000],
                "lines": result.lines[:1000],
                "corrected": result.corrected,
                "strategy": result.strategy,
                "processing_ms": result.processing_time_ms,
            }
        except ProviderError:
            raise
        except Exception as e:
            log.error("OCR recognition failed: %s", e)
            self.last_error = str(e)
            raise ProviderError("ocr_recognition_failed", f"OCR processing failed: {e}", 500)
        finally:
            if tmp_path:
                _os.unlink(tmp_path)

    def probe(self) -> dict:
        if not self.configured:
            return {"ready": False, "error": self.config_problem}
        try:
            self._load_engine()
            return {"ready": True, "error": None}
        except Exception as e:
            return {"ready": False, "error": str(e)}

    def translate(self, text: str) -> Optional[str]:
        if not text.strip():
            return None
        try:
            from controlmt import ControlMT
            if self._mt is None:
                with self._load_lock:
                    if self._mt is None:
                        log.info("Loading ControlMT translation model (first time ~280 MB)...")
                        self._mt = ControlMT.from_hf(device="cpu", quant="int8")
            return self._mt.translate(text)
        except Exception as e:
            log.warning("Translation failed: %s", e)
            return None


def clean_ocr_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


# ---------------------------------------------------------------------------
# 4. Image and PDF handling
# ---------------------------------------------------------------------------
_PDFIUM_LOCK = threading.Lock()
_UPLOAD_TYPES = {
    ".png":  ("image",  "png",  "image/png",      {"image/png"},                            ".png"),
    ".jpg":  ("image",  "jpeg", "image/jpeg",     {"image/jpeg", "image/jpg", "image/pjpeg"}, ".jpg"),
    ".jpeg": ("image",  "jpeg", "image/jpeg",     {"image/jpeg", "image/jpg", "image/pjpeg"}, ".jpg"),
    ".pdf":  ("pdf",    "pdf",  "application/pdf", {"application/pdf"},                      ".pdf"),
}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean_text_line(value: str, limit: int) -> str:
    return " ".join(_CONTROL.sub(" ", value or "").split())[:limit]


def clean_filename(name: Optional[str]) -> str:
    name = (name or "").replace("\\", "/").split("/")[-1]
    return clean_text_line(name, 255) or "document"


def sniff_type(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b"%PDF-" in data[:1024]:
        return "pdf"
    return None


def validate_upload(filename: str, content_type: Optional[str], data: bytes) -> tuple[str, str, str]:
    ext = Path(filename).suffix.lower()
    spec = _UPLOAD_TYPES.get(ext)
    if not spec:
        raise ApiError(415, "unsupported_type", "Upload a PNG, JPG, JPEG or PDF file.")
    kind, sniffed, mime, accepted, stored_ext = spec
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    declared = (content_type or " ").split(";")[0].strip().lower()
    if declared not in accepted:
        raise ApiError(415, "unsupported_mime", f"Declared type ({declared or 'unknown'}) does not match {ext}.")
    if sniff_type(data) != sniffed:
        raise ApiError(415, "content_mismatch", f"The file content is not a valid {ext} file.")
    return kind, mime, stored_ext


def read_limited(upload: UploadFile, limit: int, limit_mb: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = upload.file.read(1024 * 1024)
        if not chunk:
            break
        buf += chunk
        if len(buf) > limit:
            raise ApiError(413, "file_too_large", f"Larger than the {limit_mb} MB limit.")
    return bytes(buf)


def inspect_file(path: Path, kind: str, max_pages: int) -> int:
    try:
        if kind == "image":
            with Image.open(path) as im:
                im.load()
            return 1
        import pypdfium2 as pdfium
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(path))
            try:
                count = len(pdf)
            finally:
                pdf.close()
    except Exception as exc:
        raise ApiError(422, "corrupt_file", "This file could not be read.") from exc
    if count < 1:
        raise ApiError(422, "empty_pdf", "This PDF has no pages.")
    if count > max_pages:
        raise ApiError(413, "too_many_pages", f"This PDF has {count} pages. Limit is {max_pages}.")
    return count


def _flatten_to_rgb(im: Image.Image) -> Image.Image:
    im.load()
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return im.convert("RGB")


def render_page(path: Path, kind: str, index: int) -> Image.Image:
    if kind == "image":
        if index != 0:
            raise ApiError(404, "page_not_found", "That page does not exist.")
        with Image.open(path) as im:
            return _flatten_to_rgb(ImageOps.exif_transpose(im))
    import pypdfium2 as pdfium
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            if index < 0 or index >= len(pdf):
                raise ApiError(404, "page_not_found", "That page does not exist.")
            page = pdf[index]
            width, height = page.get_size()
            scale = max(0.5, min(4.0, OCR_MAX_SIDE / max(width, height, 1)))
            return page.render(scale=scale).to_pil().convert("RGB")
        finally:
            pdf.close()


def encode_jpeg(img: Image.Image, max_side: int, quality: int) -> bytes:
    img = img.copy()
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=quality, optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------------------
# 5. OCR pipeline
# ---------------------------------------------------------------------------
def run_pipeline(provider: ArjunaLocalProvider, path: Path, kind: str, page_count: int,
                 enhance: bool, mode: str = "auto") -> dict:
    started = time.perf_counter()
    if not provider.configured:
        err = ProviderError(
            "ocr_not_configured",
            f"{provider.config_problem} You can still type a transcription.",
            503,
        )
        return {"status": "not_processed", "ocr_text": "", "pages": [], "error": err,
                "processing_ms": 0, "attempted": False}

    pages: list[dict] = []
    texts: list[str] = []
    first_error: Optional[ProviderError] = None

    for index in range(page_count):
        number = index + 1
        page_started = time.perf_counter()
        try:
            img = render_page(path, kind, index)
            jpeg = encode_jpeg(img, OCR_MAX_SIDE, 92)
        except Exception:
            first_error = ProviderError("page_render_failed", f"Page {number} could not be read.", 422)
            pages.append({"page": number, "status": "failed", "text": "",
                          "error_code": first_error.code, "error": first_error.message})
            break
        try:
            rec = provider.recognize(jpeg, "image/jpeg", f"page-{number}.jpg",
                                     mode=mode, ensemble=enhance or True)
            text = rec["text"]
        except ProviderError as exc:
            first_error = exc
            pages.append({"page": number, "status": "failed", "text": "",
                          "error_code": exc.code, "error": exc.message})
            break
        pages.append({
            "page": number, "status": "ok", "text": text,
            "confidence": rec.get("confidence", 0.5),
            "words": rec.get("words") or [],
            "lines": rec.get("lines") or [],
            "corrected": rec.get("corrected", False),
            "strategy": rec.get("strategy", "single"),
            "ms": int((time.perf_counter() - page_started) * 1000),
        })
        if text:
            texts.append(text)

    for number in range(len(pages) + 1, page_count + 1):
        pages.append({"page": number, "status": "skipped", "text": "", "error_code": None,
                      "error": "Not processed because an earlier page failed."})

    ok = sum(1 for p in pages if p["status"] == "ok")
    status = "completed" if ok == page_count else ("partial" if ok else "failed")
    return {"status": status, "ocr_text": "\n\n".join(texts).strip(), "pages": pages, "error": first_error,
            "processing_ms": int((time.perf_counter() - started) * 1000), "attempted": True}


# ---------------------------------------------------------------------------
# 6. Storage (SQLite)
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_kind TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    provider TEXT,
    processing_ms INTEGER,
    enhance INTEGER NOT NULL DEFAULT 0,
    ocr_text TEXT NOT NULL DEFAULT '',
    edited_text TEXT,
    translation TEXT,
    page_results TEXT NOT NULL DEFAULT '[]',
    avg_confidence REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    edited_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at);
"""
_COLUMNS = {
    "title", "status", "error_code", "error_message", "provider", "processing_ms", "ocr_text",
    "edited_text", "translation", "page_results", "page_count", "edited_at", "avg_confidence",
}
_LIKE_ESCAPE = str.maketrans({"\\": "\\\\", "%": "\\%", "_": "\\_"})


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
            if "translation" not in cols:
                conn.execute("ALTER TABLE documents ADD COLUMN translation TEXT")
            if "avg_confidence" not in cols:
                conn.execute("ALTER TABLE documents ADD COLUMN avg_confidence REAL")

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def create(self, rec: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO documents (id, title, original_filename, stored_filename, mime_type, file_kind,
                       size_bytes, page_count, status, enhance, created_at, updated_at)
                   VALUES (:id, :title, :original_filename, :stored_filename, :mime_type, :file_kind,
                       :size_bytes, :page_count, :status, :enhance, :created_at, :updated_at)""", rec)

    def get(self, doc_id: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        doc = dict(row)
        doc["page_results"] = json.loads(doc["page_results"] or "[]")
        return doc

    def update(self, doc_id: str, **fields: Any) -> None:
        unknown = set(fields) - _COLUMNS
        if unknown:
            raise ValueError(f"Unknown columns: {sorted(unknown)}")
        fields["updated_at"] = now_iso()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE documents SET {assignments} WHERE id = ?", (*fields.values(), doc_id))

    def delete(self, doc_id: str) -> Optional[dict]:
        doc = self.get(doc_id)
        if doc is not None:
            with self.connect() as conn:
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return doc

    def search(self, q: Optional[str], limit: int, offset: int) -> tuple[list[dict], int]:
        where, params = "", []
        if q:
            pattern = "%" + q.translate(_LIKE_ESCAPE) + "%"
            where = ("WHERE title LIKE ? ESCAPE '\\' OR original_filename LIKE ? ESCAPE '\\' "
                     "OR ocr_text LIKE ? ESCAPE '\\' OR COALESCE(edited_text, '') LIKE ? ESCAPE '\\'")
            params = [pattern] * 4
        with self.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0]
            rows = conn.execute(
                f"""SELECT id, title, original_filename, page_count, status, created_at, updated_at,
                           (edited_text IS NOT NULL AND edited_text != ocr_text) AS is_edited,
                           substr(COALESCE(edited_text, ocr_text), 1, 200) AS snippet,
                           length(COALESCE(edited_text, ocr_text)) AS char_count,
                           avg_confidence
                    FROM documents {where} ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?""",
                (*params, limit, offset)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["is_edited"] = bool(item["is_edited"])
            item["snippet"] = " ".join((item["snippet"] or "").split())[:160]
            items.append(item)
        return items, total

    def analytics(self) -> dict:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            avg = conn.execute("SELECT AVG(avg_confidence) FROM documents WHERE avg_confidence IS NOT NULL").fetchone()[0]
            by_status = {r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")}
        return {"total": total, "avg_confidence": avg, "by_status": by_status}


def final_text(doc: dict) -> str:
    return doc["edited_text"] if doc["edited_text"] is not None else doc["ocr_text"]


def doc_public(doc: dict, include_text: bool = True) -> dict:
    out = {
        "id": doc["id"], "title": doc["title"], "original_filename": doc["original_filename"],
        "mime_type": doc["mime_type"], "file_kind": doc["file_kind"], "size_bytes": doc["size_bytes"],
        "page_count": doc["page_count"], "status": doc["status"], "error_code": doc["error_code"],
        "error": doc["error_message"], "provider": doc["provider"], "processing_ms": doc["processing_ms"],
        "enhance": bool(doc["enhance"]), "created_at": doc["created_at"], "updated_at": doc["updated_at"],
        "edited_at": doc["edited_at"],
        "is_edited": doc["edited_text"] is not None and doc["edited_text"] != doc["ocr_text"],
        "has_translation": bool(doc.get("translation")),
        "avg_confidence": doc.get("avg_confidence"),
    }
    if include_text:
        out.update(ocr_text=doc["ocr_text"], edited_text=doc["edited_text"],
                   final_text=final_text(doc), translation=doc.get("translation"), pages=doc["page_results"])
    return out


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def download_headers(title: str, ext: str) -> dict:
    unicode_name = re.sub(r'[\/:*?"<>|]', " ", clean_text_line(title, 80)).strip(" .") or "document"
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", " ",
                        unicodedata.normalize("NFKD", unicode_name).encode("ascii", "ignore").decode()).strip("._ ")
    ascii_name = ascii_name or "document"
    return {
        "Content-Disposition":
            f'attachment; filename="{ascii_name}{ext}"; filename*=UTF-8\'\'{quote(unicode_name + ext, safe="")}',
        "Cache-Control": "no-store",
    }


# ---------------------------------------------------------------------------
# 7. Evaluation (CER / WER)
# ---------------------------------------------------------------------------
def normalize_for_eval(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def edit_distance(ref, hyp) -> int:
    start = 0
    limit = min(len(ref), len(hyp))
    while start < limit and ref[start] == hyp[start]:
        start += 1
    ref, hyp = ref[start:], hyp[start:]
    end = 0
    limit = min(len(ref), len(hyp))
    while end < limit and ref[len(ref) - 1 - end] == hyp[len(hyp) - 1 - end]:
        end += 1
    if end:
        ref, hyp = ref[:len(ref) - end], hyp[:len(hyp) - end]
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    vocab: dict = {}

    def encode(seq):
        return np.fromiter((vocab.setdefault(t, len(vocab)) for t in seq), dtype=np.int32, count=len(seq))

    r, h = encode(ref), encode(hyp)
    m = len(h)
    idx = np.arange(m + 1, dtype=np.int32)
    prev = idx.copy()
    row = np.empty(m + 1, dtype=np.int32)
    for i in range(1, len(r) + 1):
        cost = (h != r[i - 1]).astype(np.int32)
        row[0] = i
        row[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        prev = np.minimum.accumulate(row - idx) + idx
    return int(prev[m])


def evaluate_texts(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize_for_eval(reference), normalize_for_eval(hypothesis)
    if not ref:
        raise ApiError(422, "empty_reference", "The reference transcription is empty.")
    if len(ref) > MAX_EVAL_CHARS or len(hyp) > MAX_EVAL_CHARS:
        raise ApiError(413, "text_too_long", f"Each text can be at most {MAX_EVAL_CHARS:,} characters.")
    ref_words, hyp_words = ref.split(), hyp.split()
    char_distance = edit_distance(ref, hyp)
    word_distance = edit_distance(ref_words, hyp_words)
    return {
        "cer": round(char_distance / len(ref), 4),
        "wer": round(word_distance / len(ref_words), 4),
        "char_edit_distance": char_distance, "reference_chars": len(ref), "hypothesis_chars": len(hyp),
        "word_edit_distance": word_distance, "reference_words": len(ref_words), "hypothesis_words": len(hyp_words),
        "reference_source": "user_supplied",
        "interpretation": "Lower is better. 0 means the OCR text matches the reference exactly.",
        "caveat": "These scores describe this one document and are only as reliable as your reference.",
        "method": "Levenshtein distance over Unicode code points (CER) and whitespace-separated words (WER).",
    }


# ---------------------------------------------------------------------------
# 8. HTTP routes and app factory
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api")


class DocumentUpdate(BaseModel):
    edited_text: Optional[str] = Field(default=None, max_length=1_000_000)
    title: Optional[str] = Field(default=None, max_length=200)


class EvaluateRequest(BaseModel):
    reference: str = Field(max_length=MAX_EVAL_CHARS * 2)
    document_id: Optional[str] = Field(default=None, max_length=64)
    hypothesis: Optional[str] = Field(default=None, max_length=MAX_EVAL_CHARS * 2)
    compare_with: Literal["ocr", "edited"] = "ocr"


def _not_found() -> ApiError:
    return ApiError(404, "document_not_found", "That document does not exist.")


def _get_or_404(request: Request, document_id: str) -> dict:
    doc = request.app.state.store.get(document_id)
    if doc is None:
        raise _not_found()
    return doc


@router.get("/health")
def health(request: Request):
    settings = request.app.state.settings
    db_ok = request.app.state.store.ping()
    return {
        "status": "ok" if db_ok else "degraded",
        "version": VERSION,
        "database": "ok" if db_ok else "error",
        "time": now_iso(),
        "limits": {"max_upload_mb": settings.max_upload_mb, "max_pdf_pages": settings.max_pdf_pages,
                   "allowed_extensions": [".png", ".jpg", ".jpeg", ".pdf"]},
        "storage": {"uploads": "data/uploads/", "database": "data/kannadasetu.db"},
        "translation": "local-controlmt",
    }


@router.get("/provider-status")
def provider_status(request: Request, probe: bool = False):
    provider = request.app.state.provider
    problem = provider.config_problem
    out = {
        "provider": provider.name,
        "configured": problem is None,
        "problem": problem,
        "last_success_at": provider.last_success_at,
        "last_error": provider.last_error,
        "live_inference_verified": provider.last_success_at is not None,
        "translation": "local-controlmt",
        "note": ("First OCR run downloads ~170 MB of models and can take a minute."
                 if problem is None else problem),
    }
    if probe:
        out["probe"] = provider.probe()
    return out


@router.post("/ocr", status_code=201)
def run_ocr(request: Request, file: UploadFile = File(...), title: Optional[str] = Form(None),
            enhance: bool = Form(False), handwriting: bool = Form(False)):
    state = request.app.state
    settings, store, provider = state.settings, state.store, state.provider
    filename = clean_filename(file.filename)
    data = read_limited(file, settings.max_upload_bytes, settings.max_upload_mb)
    kind, mime, ext = validate_upload(filename, file.content_type, data)
    doc_id = uuid.uuid4().hex
    stored = doc_id + ext
    path = settings.upload_dir / stored
    path.write_bytes(data)
    try:
        page_count = inspect_file(path, kind, settings.max_pdf_pages)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    now = now_iso()
    store.create({
        "id": doc_id, "title": clean_text_line(title or "", 200) or (Path(filename).stem or filename)[:120],
        "original_filename": filename, "stored_filename": stored, "mime_type": mime, "file_kind": kind,
        "size_bytes": len(data), "page_count": page_count, "status": "processing",
        "enhance": int(enhance or handwriting),
        "created_at": now, "updated_at": now,
    })
    mode = "handwriting" if handwriting else "auto"
    try:
        result = run_pipeline(provider, path, kind, page_count, enhance or handwriting, mode=mode)
    except Exception:
        store.update(doc_id, status="failed", error_code="internal_error",
                     error_message="Unexpected error while processing the document.")
        log.error("OCR pipeline crashed for document %s", doc_id)
        raise ApiError(500, "internal_error", "Unexpected error while processing the document.",
                       document_id=doc_id)
    err: Optional[ProviderError] = result["error"]
    # Compute average confidence
    confs = [p.get("confidence") for p in result["pages"] if p.get("status") == "ok" and p.get("confidence") is not None]
    avg_conf = float(np.mean(confs)) if confs else None
    store.update(
        doc_id, status=result["status"], ocr_text=result["ocr_text"],
        page_results=json.dumps(result["pages"], ensure_ascii=False),
        error_code=err.code if err else None, error_message=err.message if err else None,
        provider=provider.name if result["attempted"] else None,
        processing_ms=result["processing_ms"], avg_confidence=avg_conf)
    log.info("document=%s pages=%d status=%s ms=%d", doc_id, page_count, result["status"], result["processing_ms"])
    if result["status"] in ("failed", "not_processed") and err:
        raise ApiError(err.status, err.code, err.message, document_id=doc_id, retry_after=err.retry_after)
    return doc_public(store.get(doc_id))


@router.post("/documents/{document_id}/translate")
def translate_document(document_id: str, request: Request):
    provider = request.app.state.provider
    doc = _get_or_404(request, document_id)
    text = final_text(doc)
    if not text.strip():
        raise ApiError(422, "no_text", "This document has no text to translate.")
    translated = provider.translate(text)
    if not translated:
        raise ApiError(502, "translation_failed", "The translation model did not respond.")
    request.app.state.store.update(document_id, translation=translated)
    return {"translation": translated}


@router.post("/documents/{document_id}/correct")
def correct_word(document_id: str, request: Request, original: str = Form(...),
                 corrected: str = Form(...), context: str = Form("")):
    _get_or_404(request, document_id)
    ce = request.app.state.correction_engine
    ce.add_correction(original, corrected, context, document_id)
    return {"suggestions": ce.get_suggestions(original)}


@router.get("/suggestions")
def suggestions(request: Request, word: str = Query(...)):
    ce = request.app.state.correction_engine
    return {"word": word, "suggestions": ce.get_suggestions(word)}


@router.get("/analytics/common-errors")
def analytics_errors(request: Request, limit: int = Query(20, ge=1, le=100)):
    ce = request.app.state.correction_engine
    return {"errors": ce.get_common_errors(limit)}


@router.get("/analytics/summary")
def analytics_summary(request: Request):
    store = request.app.state.store
    ce = request.app.state.correction_engine
    summary = store.analytics()
    errors = ce.get_common_errors(10)
    return {
        "total_documents": summary["total"],
        "average_confidence": summary["avg_confidence"],
        "by_status": summary["by_status"],
        "top_corrections": errors,
        "top_corrections_total": sum(e["count"] for e in errors),
    }


@router.get("/documents")
def list_documents(request: Request, q: Optional[str] = Query(None, max_length=200),
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    items, total = request.app.state.store.search((q or "").strip() or None, limit, offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/documents/{document_id}")
def get_document(document_id: str, request: Request):
    return doc_public(_get_or_404(request, document_id))


@router.put("/documents/{document_id}")
def update_document(document_id: str, body: DocumentUpdate, request: Request):
    store = request.app.state.store
    _get_or_404(request, document_id)
    fields: dict = {}
    given = body.model_fields_set
    if "title" in given:
        title = clean_text_line(body.title or "", 200)
        if not title:
            raise ApiError(422, "invalid_title", "The title cannot be empty.")
        fields["title"] = title
    if "edited_text" in given:
        if body.edited_text is None:
            fields.update(edited_text=None, edited_at=None)
        else:
            fields.update(edited_text=normalize_newlines(body.edited_text), edited_at=now_iso())
    if not fields:
        raise ApiError(400, "nothing_to_update", "Send edited_text and/or title.")
    store.update(document_id, **fields)
    return doc_public(store.get(document_id))


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request):
    doc = request.app.state.store.delete(document_id)
    if doc is None:
        raise _not_found()
    (request.app.state.settings.upload_dir / doc["stored_filename"]).unlink(missing_ok=True)
    return Response(status_code=204)


@router.get("/documents/{document_id}/pages/{page}")
def document_page(document_id: str, page: int, request: Request):
    doc = _get_or_404(request, document_id)
    path = request.app.state.settings.upload_dir / doc["stored_filename"]
    if not path.is_file():
        raise ApiError(404, "file_missing", "The original file is no longer on disk.")
    if page < 1 or page > doc["page_count"]:
        raise ApiError(404, "page_not_found", "That page does not exist.")
    try:
        jpeg = encode_jpeg(render_page(path, doc["file_kind"], page - 1), PREVIEW_MAX_SIDE, 85)
    except ApiError:
        raise
    except Exception:
        raise ApiError(422, "page_render_failed", "This page could not be rendered.")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"})


@router.get("/documents/{document_id}/export.txt")
def export_txt(document_id: str, request: Request):
    doc = _get_or_404(request, document_id)
    return Response(content=final_text(doc).encode("utf-8"), media_type="text/plain; charset=utf-8",
                    headers=download_headers(doc["title"], ".txt"))


@router.get("/documents/{document_id}/export.json")
def export_json(document_id: str, request: Request):
    doc = _get_or_404(request, document_id)
    public = doc_public(doc)
    payload = {
        "application": "KannadaSetu AI", "format_version": 2, "encoding": "utf-8", "exported_at": now_iso(),
        "document": {k: public.get(k) for k in ("id", "title", "original_filename", "mime_type", "page_count",
                                                "status", "provider", "processing_ms", "created_at", "updated_at",
                                                "edited_at", "avg_confidence")},
        "text": {"ocr": doc["ocr_text"], "edited": doc["edited_text"], "final": final_text(doc),
                 "is_edited": public["is_edited"], "translation": doc.get("translation")},
        "pages": [{k: p.get(k) for k in ("page", "status", "text", "confidence", "error")}
                  for p in doc["page_results"]],
    }
    return Response(content=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    media_type="application/json", headers=download_headers(doc["title"], ".json"))


@router.get("/documents/{document_id}/export.docx")
def export_docx(document_id: str, request: Request):
    doc = _get_or_404(request, document_id)
    engine = ExportEngine()
    try:
        data = engine.export_docx(final_text(doc), doc["title"])
    except ImportError:
        raise ApiError(500, "export_unavailable", "python-docx is not installed.")
    return Response(content=data, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers=download_headers(doc["title"], ".docx"))


@router.get("/documents/{document_id}/export.pdf")
def export_pdf(document_id: str, request: Request):
    doc = _get_or_404(request, document_id)
    engine = ExportEngine()
    try:
        data = engine.export_pdf(final_text(doc), doc["title"])
    except ImportError:
        raise ApiError(500, "export_unavailable", "reportlab is not installed.")
    return Response(content=data, media_type="application/pdf",
                    headers=download_headers(doc["title"], ".pdf"))


@router.post("/evaluate")
def evaluate(body: EvaluateRequest, request: Request):
    if body.document_id:
        doc = _get_or_404(request, body.document_id)
        if body.compare_with == "edited":
            hypothesis, compared = final_text(doc), "edited_text"
        else:
            hypothesis, compared = doc["ocr_text"], "ocr_text"
        if not hypothesis.strip():
            raise ApiError(422, "no_ocr_text", "This document has no OCR output to evaluate.")
    elif body.hypothesis is not None:
        hypothesis, compared = body.hypothesis, "pasted_text"
    else:
        raise ApiError(422, "missing_hypothesis", "Choose a saved document or paste the OCR text to compare.")
    result = evaluate_texts(body.reference, hypothesis)
    result["compared"] = compared
    result["document_id"] = body.document_id
    return result


@router.post("/sample", status_code=201)
def run_sample(request: Request):
    """Generate a synthetic Kannada sample and OCR it (used by the demo button)."""
    from PIL import ImageDraw, ImageFont
    settings = request.app.state.settings
    doc_id = uuid.uuid4().hex
    img = Image.new("RGB", (1000, 340), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
        str(BASE_DIR / "fonts" / "NotoSansKannada-Regular.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansKannada-Regular.ttf",
    ):
        if Path(candidate).exists():
            try:
                font = ImageFont.truetype(candidate, 44)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    draw.text((40, 60), "ಕನ್ನಡ ಸೇತು", fill="black", font=font)
    draw.text((40, 160), "ಇದು ಒಂದು ಮಾದರಿ ದಾಖಲೆ.", fill="black", font=font)
    path = settings.upload_dir / f"{doc_id}.png"
    img.save(path)
    now = now_iso()
    request.app.state.store.create({
        "id": doc_id, "title": "Sample Kannada document",
        "original_filename": "sample.png", "stored_filename": path.name,
        "mime_type": "image/png", "file_kind": "image", "size_bytes": path.stat().st_size,
        "page_count": 1, "status": "processing", "enhance": 1,
        "created_at": now, "updated_at": now,
    })
    try:
        result = run_pipeline(request.app.state.provider, path, "image", 1, True, mode="auto")
        err = result["error"]
        confs = [p.get("confidence") for p in result["pages"]
                 if p.get("status") == "ok" and p.get("confidence") is not None]
        avg_conf = float(np.mean(confs)) if confs else None
        request.app.state.store.update(
            doc_id, status=result["status"], ocr_text=result["ocr_text"],
            page_results=json.dumps(result["pages"], ensure_ascii=False),
            error_code=err.code if err else None,
            error_message=err.message if err else None,
            provider=request.app.state.provider.name if result["attempted"] else None,
            processing_ms=result["processing_ms"], avg_confidence=avg_conf,
        )
    except Exception as exc:
        request.app.state.store.update(
            doc_id, status="failed", error_code="internal_error", error_message=str(exc),
        )
    return doc_public(request.app.state.store.get(doc_id))


_PAGE_CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline' "
             "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; "
             "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="KannadaSetu AI", version=VERSION,
        description="Digitise Kannada documents locally with arjuna-ocr. "
                    "Kannada→English translation runs locally via ControlMT. No external APIs.")
    app.state.settings = settings
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = Store(settings.db_path)
    app.state.store.init()
    app.state.provider = ArjunaLocalProvider(settings)
    app.state.correction_engine = CorrectionEngine(settings.data_dir)
    app.state.export_engine = ExportEngine()

    def _warm():
        try:
            KannadaOCREngine().warmup()
        except Exception as exc:
            log.info("Warmup skipped: %s", exc)
    threading.Thread(target=_warm, daemon=True).start()

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        headers = {}
        retry_after = exc.extra.get("retry_after")
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        body = {"code": exc.code, "message": exc.message}
        body.update({k: v for k, v in exc.extra.items() if v is not None})
        return JSONResponse({"error": body}, status_code=exc.status, headers=headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse({"error": {"code": f"http_{exc.status_code}", "message": message}},
                            status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in err.get("loc", ())[1:]), "message": err.get("msg", "Invalid")}
                   for err in exc.errors()]
        first = details[0] if details else {"field": "", "message": "Invalid request."}
        label = f"{first['field']}: " if first["field"] else ""
        return JSONResponse({"error": {"code": "validation_error", "message": f"{label}{first['message']}",
                                       "details": details}}, status_code=422)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("Unhandled %s on %s", type(exc).__name__, request.url.path)
        return JSONResponse({"error": {"code": "internal_error", "message": "Unexpected server error."}},
                            status_code=500)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(BASE_DIR / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-cache", "Content-Security-Policy": _PAGE_CSP})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    return app


app = create_app()