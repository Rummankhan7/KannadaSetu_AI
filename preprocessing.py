"""Advanced preprocessing pipeline for Kannada OCR.

Multi-strategy preprocessing tuned for both printed and handwritten Kannada.
Every strategy is CPU-only and deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("kannadasetu.preprocessing")


@dataclass
class PreprocessResult:
    image: Image.Image
    strategy: str
    deskew_angle: float


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def preprocess_for_ocr(image: Image.Image, mode: str = "auto") -> Image.Image:
    return preprocess_variants(image, mode=mode, max_variants=1)[0].image


def preprocess_variants(
    image: Image.Image, mode: str = "auto", max_variants: int = 3
) -> List[PreprocessResult]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = ImageOps.exif_transpose(image)
    deskewed, angle = _deskew(image)

    if mode == "auto":
        mode = detect_mode(deskewed)

    if mode == "handwriting":
        strategies = [_hw_clahe_sauvola, _hw_adaptive, _hw_otsu_dilate]
    else:
        strategies = [_print_adaptive, _print_otsu, _print_clahe]

    results: List[PreprocessResult] = []
    for fn in strategies[:max_variants]:
        try:
            out = fn(deskewed)
            results.append(PreprocessResult(out, fn.__name__, angle))
        except Exception as exc:
            log.warning("Strategy %s failed: %s", fn.__name__, exc)
    if not results:
        results.append(PreprocessResult(deskewed, "fallback", angle))
    return results


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------
def detect_mode(image: Image.Image) -> str:
    try:
        gray = np.array(image.convert("L"))
        var = calculate_stroke_variance(gray)
        return "handwriting" if var > 15.0 else "printed"
    except Exception:
        return "printed"


# Kept for backward-compat import in existing code paths
_detect_mode = detect_mode


def calculate_stroke_variance(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    dist = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 0.0
    return float(np.var(vals))


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------
def _deskew(image: Image.Image, max_angle: float = 15.0) -> tuple[Image.Image, float]:
    try:
        gray = np.array(image.convert("L"))
        inv = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inv > 0))
        if coords.shape[0] < 50:
            return image, 0.0
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.5 or abs(angle) > max_angle:
            return image, 0.0
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(image), M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated), float(angle)
    except Exception:
        return image, 0.0


# ---------------------------------------------------------------------------
# Strategy: printed
# ---------------------------------------------------------------------------
def _print_adaptive(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    return Image.fromarray(denoised).convert("RGB")


def _print_otsu(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


def _print_clahe(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(clahe).convert("RGB")


# ---------------------------------------------------------------------------
# Strategy: handwriting
# ---------------------------------------------------------------------------
def _hw_clahe_sauvola(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    binary = _sauvola(clahe, window_size=31, k=0.34)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    denoised = cv2.medianBlur(cleaned, 3)
    return Image.fromarray(denoised).convert("RGB")


def _hw_adaptive(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    binary = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15
    )
    return Image.fromarray(binary).convert("RGB")


def _hw_otsu_dilate(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 30)
    norm = cv2.divide(gray, blur, scale=255)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    return Image.fromarray(dilated).convert("RGB")


def _sauvola(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window_size, window_size))
    mean_sq = cv2.blur(gray_f * gray_f, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    threshold = mean * (1 + k * (std / 128 - 1))
    return ((gray_f > threshold).astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Public helpers kept for compatibility
# ---------------------------------------------------------------------------
def enhance_for_handwriting(image: Image.Image) -> Image.Image:
    return _hw_clahe_sauvola(image)


def enhance_for_printed(image: Image.Image) -> Image.Image:
    return _print_adaptive(image)


def deskew_image(image: Image.Image) -> Image.Image:
    return _deskew(image)[0]


def sauvola_binarization(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    return _sauvola(gray, window_size, k)