"""Interactive correction engine for OCR results + analytics."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("kannadasetu.correction")


@dataclass
class Correction:
    original: str
    corrected: str
    context: str
    document_id: str
    timestamp: str


class CorrectionEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_file = self.data_dir / "corrections.json"
        self._lock = threading.Lock()
        self.corrections: List[Correction] = []
        self._load_corrections()

    def _load_corrections(self) -> None:
        if not self.corrections_file.exists():
            return
        try:
            raw = json.loads(self.corrections_file.read_text(encoding="utf-8"))
            self.corrections = [Correction(**c) for c in raw if isinstance(c, dict)]
            log.info("Loaded %d corrections", len(self.corrections))
        except Exception as exc:
            log.error("Failed to load corrections: %s", exc)

    def _save_corrections(self) -> None:
        try:
            tmp = self.corrections_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps([asdict(c) for c in self.corrections], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.corrections_file)
        except Exception as exc:
            log.error("Failed to save corrections: %s", exc)

    def add_correction(self, original: str, corrected: str, context: str = "", document_id: str = "") -> None:
        if not original or original == corrected:
            return
        with self._lock:
            self.corrections.append(
                Correction(
                    original=original,
                    corrected=corrected,
                    context=context[:200],
                    document_id=document_id,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
            )
            if len(self.corrections) > 50_000:
                self.corrections = self.corrections[-40_000:]
            self._save_corrections()

    def get_suggestions(self, word: str, context: str = "", limit: int = 5) -> List[str]:
        if not word:
            return []
        counts: Dict[str, int] = {}
        with self._lock:
            for c in self.corrections:
                if c.original == word:
                    counts[c.corrected] = counts.get(c.corrected, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]]

    def get_common_errors(self, limit: int = 20) -> List[Dict]:
        agg: Dict[str, Dict] = {}
        with self._lock:
            for c in self.corrections:
                entry = agg.setdefault(c.original, {"original": c.original, "corrections": {}, "count": 0})
                entry["count"] += 1
                entry["corrections"][c.corrected] = entry["corrections"].get(c.corrected, 0) + 1
        out = []
        for entry in sorted(agg.values(), key=lambda e: e["count"], reverse=True)[:limit]:
            top = max(entry["corrections"].items(), key=lambda kv: kv[1])[0]
            out.append({"original": entry["original"], "corrected": top, "count": entry["count"]})
        return out