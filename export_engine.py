"""Export engine for KannadaSetu AI — TXT, JSON, DOCX, PDF."""
import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict

log = logging.getLogger("kannadasetu.export")


class ExportEngine:
    def export_txt(self, text: str, title: str) -> bytes:
        return text.encode("utf-8")

    def export_json(self, document: Dict) -> bytes:
        payload = {
            "application": "KannadaSetu AI",
            "format_version": 2,
            "encoding": "utf-8",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "document": document,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    def export_docx(self, text: str, title: str) -> bytes:
        try:
            from docx import Document
        except ImportError:
            log.error("python-docx not installed")
            raise
        doc = Document()
        doc.add_heading(title, level=1)
        doc.add_paragraph(
            f"Exported from KannadaSetu AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        doc.add_paragraph("")
        for para in text.split("\n"):
            if para.strip():
                doc.add_paragraph(para)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def export_pdf(self, text: str, title: str) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import cm
        except ImportError:
            log.error("reportlab not installed")
            raise
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm, title)
        c.setFont("Helvetica", 10)
        c.drawString(
            2 * cm,
            height - 3 * cm,
            f"Exported from KannadaSetu AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        c.setFont("Helvetica", 12)
        y = height - 5 * cm
        for line in text.split("\n"):
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 12)
            c.drawString(2 * cm, y, line)
            y -= 0.5 * cm
        c.save()
        return buf.getvalue()