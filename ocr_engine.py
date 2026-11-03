"""Advanced OCR Engine for KannadaSetu AI — v3.1

Robust to arjuna-ocr API variations:
- Tries native per-word confidence from the model first.
- Falls back to synthesizing per-word confidence from text quality heuristics
  and ensemble agreement, so the UI always has word-level data.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("kannadasetu.ocr")


_KANNADA_CONFUSIONS: Dict[str, List[str]] = {
    "ಣ": ["ನ"], "ನ": ["ಣ"], "ರ": ["ಱ"], "ಷ": ["ಷ"], "ೀ": ["ಿ"], "ೂ": ["ು"],
    "ೆ": ["ೇ"], "ೇ": ["ೆ"],
}

_KANNADA_WORDS = {
    "ಮತ್ತು", "ಇದು", "ಆದರೆ", "ಎಂದು", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಅಲ್ಲಿ", "ಅವರು",
    "ನಾನು", "ನೀವು", "ನಮ್ಮ", "ನಿಮ್ಮ", "ಅವರ", "ಇವರ", "ಆದ್ದರಿಂದ", "ಹಾಗೂ",
    "ಕನ್ನಡ", "ಭಾಷೆ", "ದೇಶ", "ರಾಜ್ಯ", "ಸರ್ಕಾರ", "ಜನರು", "ವರ್ಷ", "ದಿನ",
    "ಸಮಯ", "ಕೆಲಸ", "ಮನೆ", "ನೀರು", "ಆಹಾರ", "ಶಾಲೆ", "ವಿದ್ಯಾರ್ಥಿ", "ಶಿಕ್ಷಕ",
    "ಒಂದು", "ಮಾದರಿ", "ದಾಖಲೆ", "ಸೇತು",
}

_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]+")
_WORD_RE = re.compile(r"\S+")


@dataclass
class OCRResult:
    text: str
    confidence: float
    words: List[Dict] = field(default_factory=list)
    lines: List[Dict] = field(default_factory=list)
    blocks: List[Dict] = field(default_factory=list)
    processing_time_ms: int = 0
    model_used: str = "arjuna-ocr"
    strategy: str = "single"
    corrected: bool = False


class KannadaOCREngine:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._ocr = None
        self._load_lock = threading.Lock()
        self.models_loaded = False
        self._warm = False

    def _load_models(self) -> None:
        if self.models_loaded:
            return
        with self._load_lock:
            if self.models_loaded:
                return
            log.info("Loading Arjuna-OCR model...")
            from kanen_infer import KanEnOCR  # noqa
            self._ocr = KanEnOCR()
            self.models_loaded = True
            log.info("Arjuna-OCR model ready")

    def warmup(self) -> bool:
        if self._warm:
            return True
        try:
            self._load_models()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                Image.new("RGB", (64, 64), "white").save(tmp.name)
                path = tmp.name
            try:
                _ = self._ocr.page(path)
            finally:
                os.unlink(path)
            self._warm = True
            return True
        except Exception as exc:
            log.warning("Warmup failed: %s", exc)
            return False

    def recognize(
        self,
        image: Image.Image,
        mode: str = "auto",
        enhance: bool = True,
        ensemble: bool = True,
        correct: bool = True,
    ) -> OCRResult:
        from preprocessing import preprocess_variants, detect_mode

        started = time.perf_counter()
        self._load_models()

        if mode == "auto":
            mode = detect_mode(image)
            log.info("Detected mode: %s", mode)

        if enhance or ensemble:
            variants = preprocess_variants(image, mode=mode, max_variants=3)
        else:
            class _V:
                def __init__(self, im):
                    self.image = im
                    self.strategy = "raw"
                    self.deskew_angle = 0.0
            variants = [_V(image)]

        results: List[OCRResult] = []
        for v in variants:
            try:
                r = self._run_single(v.image)
                r.strategy = v.strategy
                results.append(r)
            except Exception as exc:
                log.warning("Variant %s failed: %s", getattr(v, "strategy", "?"), exc)

        if not results:
            return OCRResult(text="", confidence=0.0, strategy="failed")

        # Pick the best-scoring variant
        best = max(
            results,
            key=lambda r: r.confidence * np.log(len(r.text) + 2),
        )

        # If the best variant didn't have real per-word confidence,
        # synthesize words + confidence from the text and ensemble agreement.
        if not best.words and best.text.strip():
            best.words, best.lines, best.blocks = _synthesize_words_from_text(
                best.text, ensemble_texts=[r.text for r in results]
            )
            # Recompute overall confidence now that we have words
            best.confidence = _weighted_confidence(best.words, best.text)

        if correct and best.text:
            corrected_text, did_correct = _lexicon_correct(best.text, best.words)
            if did_correct:
                best.text = corrected_text
                best.corrected = True

        best.processing_time_ms = int((time.perf_counter() - started) * 1000)
        return best

    def _run_single(self, image: Image.Image) -> OCRResult:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            path = tmp.name
        try:
            doc = self._ocr.page(path)
            text = self._safe_render(doc)
            words, lines, blocks = _parse_structure(doc)
            confidence = _weighted_confidence(words, text)
            return OCRResult(
                text=text, confidence=confidence,
                words=words, lines=lines, blocks=blocks,
            )
        finally:
            os.unlink(path)

    def _safe_render(self, doc) -> str:
        """Try every known arjuna-ocr render signature + direct attribute access."""
        # 1. Try render with 'txt' arg
        try:
            out = self._ocr.render(doc, "txt")
            if out and str(out).strip():
                return str(out).strip()
        except Exception as e:
            log.debug("render(doc, 'txt') failed: %s", e)

        # 2. Try render without args
        try:
            out = self._ocr.render(doc)
            if out and str(out).strip():
                return str(out).strip()
        except Exception as e:
            log.debug("render(doc) failed: %s", e)

        # 3. Try kwargs
        try:
            out = self._ocr.render(doc, format="txt")
            if out and str(out).strip():
                return str(out).strip()
        except Exception as e:
            log.debug("render(doc, format='txt') failed: %s", e)

        # 4. Direct attribute access on the doc object
        for attr in ("text", "content", "transcription", "raw_text"):
            val = getattr(doc, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # 5. Dict access
        if isinstance(doc, dict):
            for key in ("text", "content", "transcription", "raw_text"):
                val = doc.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        # 6. If doc is a string itself
        if isinstance(doc, str) and doc.strip():
            return doc.strip()

        # 7. Nothing worked
        log.warning("Could not extract text from doc of type %s", type(doc).__name__)
        return ""

    def recognize(
        self,
        image: Image.Image,
        mode: str = "auto",
        enhance: bool = True,
        ensemble: bool = True,
        correct: bool = True,
    ) -> OCRResult:
        from preprocessing import preprocess_variants, detect_mode

        started = time.perf_counter()
        self._load_models()

        if mode == "auto":
            mode = detect_mode(image)
            log.info("Detected mode: %s", mode)

        if enhance or ensemble:
            variants = preprocess_variants(image, mode=mode, max_variants=3)
        else:
            class _V:
                def __init__(self, im):
                    self.image = im
                    self.strategy = "raw"
                    self.deskew_angle = 0.0
            variants = [_V(image)]

        results: List[OCRResult] = []
        for v in variants:
            try:
                r = self._run_single(v.image)
                r.strategy = v.strategy
                if r.text.strip():
                    results.append(r)
                else:
                    log.warning("Variant %s returned empty text", v.strategy)
            except Exception as exc:
                log.warning("Variant %s failed: %s", getattr(v, "strategy", "?"), exc)

        if not results:
            log.error("All OCR variants returned empty text")
            return OCRResult(text="", confidence=0.0, strategy="failed")

        best = max(results, key=lambda r: r.confidence * np.log(len(r.text) + 2))

        # Force synthesis if the model didn't return per-word data
        if not best.words and best.text.strip():
            log.info("Synthesizing per-word confidence for %d chars", len(best.text))
            best.words, best.lines, best.blocks = _synthesize_words_from_text(
                best.text, ensemble_texts=[r.text for r in results]
            )
            best.confidence = _weighted_confidence(best.words, best.text)
            log.info(
                "Synthesized %d words, avg confidence %.3f",
                len(best.words), best.confidence,
            )

        if correct and best.text:
            corrected_text, did_correct = _lexicon_correct(best.text, best.words)
            if did_correct:
                best.text = corrected_text
                best.corrected = True

        best.processing_time_ms = int((time.perf_counter() - started) * 1000)
        return best


# ---------------------------------------------------------------------------
# Structure parsing (native confidence path)
# ---------------------------------------------------------------------------
def _parse_structure(doc) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    words: List[Dict] = []
    lines: List[Dict] = []
    blocks: List[Dict] = []

    def get(node, key, default=None):
        if isinstance(node, dict):
            return node.get(key, default)
        return getattr(node, key, default)

    top_blocks = get(doc, "blocks", None)
    if not top_blocks:
        return words, lines, blocks

    for block in top_blocks or []:
        block_lines = []
        block_bbox = tuple(get(block, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
        for line in get(block, "lines", []) or []:
            line_words = []
            line_bbox = tuple(get(line, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
            for word in get(line, "words", []) or []:
                wtext = get(word, "text", "") or ""
                conf_raw = get(word, "confidence", None)
                conf = float(conf_raw) if conf_raw is not None else 0.5
                if not (0.0 <= conf <= 1.0):
                    conf = 0.5
                bbox = tuple(get(word, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))
                rec = {"text": wtext, "confidence": conf, "bbox": list(bbox)}
                words.append(rec)
                line_words.append(rec)
            if line_words:
                line_text = " ".join(w["text"] for w in line_words)
                lines.append(
                    {
                        "text": line_text,
                        "confidence": float(np.mean([w["confidence"] for w in line_words])),
                        "bbox": list(line_bbox),
                    }
                )
                block_lines.append(line_text)
        if block_lines:
            confs = [l["confidence"] for l in lines[-len(block_lines):]] or [0.5]
            blocks.append(
                {
                    "text": "\n".join(block_lines),
                    "confidence": float(np.mean(confs)),
                    "bbox": list(block_bbox),
                    "type": get(block, "type", "text"),
                }
            )
    return words, lines, blocks


# ---------------------------------------------------------------------------
# Synthesize per-word confidence when the model returns only plain text
# ---------------------------------------------------------------------------
def _synthesize_words_from_text(
    text: str, ensemble_texts: Optional[List[str]] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build word-level confidence from text quality + ensemble agreement."""
    words: List[Dict] = []
    lines: List[Dict] = []
    blocks: List[Dict] = []

    if not text.strip():
        return words, lines, blocks

    # Count how often each word appears across ensemble variants.
    # Words that only appear in the winner variant get a small boost;
    # words that appear in multiple variants get a higher confidence.
    variant_word_counts: Dict[str, int] = {}
    if ensemble_texts:
        for t in ensemble_texts:
            seen = set(t.split())
            for w in seen:
                variant_word_counts[w] = variant_word_counts.get(w, 0) + 1
    total_variants = max(1, len(ensemble_texts or []))

    # Group into lines (as the model produced them)
    for line_idx, raw_line in enumerate(text.split("\n")):
        stripped = raw_line.strip()
        if not stripped:
            continue
        line_words: List[Dict] = []
        for tok in stripped.split():
            conf = _word_confidence(tok, variant_word_counts, total_variants)
            rec = {
                "text": tok,
                "confidence": conf,
                "bbox": [0, 0, 0, 0],  # no bbox available in fallback
            }
            words.append(rec)
            line_words.append(rec)
        if line_words:
            lines.append(
                {
                    "text": stripped,
                    "confidence": float(np.mean([w["confidence"] for w in line_words])),
                    "bbox": [0, 0, 0, 0],
                }
            )
    if lines:
        blocks.append(
            {
                "text": "\n".join(l["text"] for l in lines),
                "confidence": float(np.mean([l["confidence"] for l in lines])),
                "bbox": [0, 0, 0, 0],
                "type": "text",
            }
        )
    return words, lines, blocks


def _word_confidence(
    word: str, variant_counts: Dict[str, int], total_variants: int
) -> float:
    """Heuristic per-word confidence in [0.25, 0.98]."""
    base = 0.55

    # 1. Kannada content ratio: words with actual Kannada chars score higher
    kn_chars = len(_KANNADA_RE.findall(word))
    if len(word) > 0:
        kn_ratio = kn_chars / len(word)
        base += kn_ratio * 0.15  # up to +0.15

    # 2. Known lexicon words get a big boost (they're real Kannada words)
    if word.strip(".,;:!?\"'()[]") in _KANNADA_WORDS:
        base += 0.20

    # 3. Ensemble agreement: words appearing in multiple variants are more reliable
    agreement = variant_counts.get(word, 0) / max(1, total_variants)
    base += agreement * 0.15

    # 4. Length: extremely short tokens (1 char) are noisier
    if len(word) <= 1:
        base -= 0.15

    # 5. Digits and ASCII-heavy tokens get a small boost (usually correct)
    if word.isascii() and any(c.isdigit() for c in word):
        base += 0.10

    return float(max(0.25, min(0.98, base)))


# ---------------------------------------------------------------------------
# Confidence + correction helpers
# ---------------------------------------------------------------------------
def _weighted_confidence(words: List[Dict], text: str) -> float:
    if words:
        total = sum(len(w["text"]) for w in words) or 1
        return float(sum(len(w["text"]) * w["confidence"] for w in words) / total)
    if not text:
        return 0.0
    kn_chars = len(_KANNADA_RE.findall(text))
    ratio = kn_chars / max(1, len(text))
    return float(min(0.9, 0.3 + ratio * 0.6))


def _lexicon_correct(text: str, words: List[Dict]) -> Tuple[str, bool]:
    if not text:
        return text, False
    low_conf_words = {
        w["text"]
        for w in words
        if w.get("confidence", 1.0) < 0.55 and _KANNADA_RE.search(w.get("text", ""))
    }
    if not low_conf_words:
        return text, False

    changed = False
    out = text
    for w in low_conf_words:
        if w in _KANNADA_WORDS:
            continue
        for i, ch in enumerate(w):
            for cand in _KANNADA_CONFUSIONS.get(ch, []):
                if cand == ch:
                    continue
                candidate = w[:i] + cand + w[i + 1:]
                if candidate in _KANNADA_WORDS:
                    out = out.replace(w, candidate)
                    changed = True
                    break
            if changed:
                break
    return out, changed