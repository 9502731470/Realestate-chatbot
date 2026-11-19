"""Section-aware chunking with contextual headers.

Every chunk is prefixed with a short header naming the record it came from
(title, location, source, kind). This "contextual chunk header" technique
markedly improves retrieval for chunks that would otherwise be ambiguous
("Completion is scheduled for December 2029" -> *which* project?).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Record

_SENTENCE = re.compile(r"(?<=[.!?؟])\s+")

SINGLE_CHUNK_KINDS = {"listing", "stat", "faq"}


@dataclass(frozen=True)
class Chunk:
    record_id: str
    ord: int
    title: str
    text: str


def _header(rec: Record) -> str:
    bits = [rec.title]
    loc = rec.location_label()
    if loc and rec.kind in ("listing", "project") and loc.lower() not in rec.title.lower():
        bits.append(loc)
    return f"{' — '.join(bits)} [{rec.source_label} {rec.kind}]"


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title, buf = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                sections.append((current_title, "\n".join(buf).strip()))
            current_title, buf = line[3:].strip(), []
        elif line.startswith("# ") and not sections and not buf:
            continue  # document title is already in the header
        else:
            buf.append(line)
    if buf:
        sections.append((current_title, "\n".join(buf).strip()))
    return [(t, b) for t, b in sections if b]


def _windows(paragraphs: list[str], max_words: int, overlap: int) -> list[str]:
    units: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            units.append(para)
            continue
        sentences = _SENTENCE.split(para)
        for sentence in sentences:
            sw = sentence.split()
            while len(sw) > max_words:
                units.append(" ".join(sw[:max_words]))
                sw = sw[max_words - overlap :]
            if sw:
                units.append(" ".join(sw))

    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for unit in units:
        n = len(unit.split())
        if buf and count + n > max_words:
            chunks.append("\n".join(buf))
            tail = " ".join(" ".join(buf).split()[-overlap:]) if overlap else ""
            buf, count = ([tail] if tail else []), len(tail.split())
        buf.append(unit)
        count += n
    if buf:
        chunks.append("\n".join(buf))
    return chunks


# ~110 words keeps a chunk (plus its header) close to the embedding model's
# 128-token window, so the vector represents the whole chunk instead of a
# truncated prefix. Smaller chunks also mean fewer prompt tokens per citation.
def chunk_record(rec: Record, max_words: int = 110, overlap_words: int = 20) -> list[Chunk]:
    header = _header(rec)
    if rec.kind in SINGLE_CHUNK_KINDS or len(rec.text.split()) <= max_words:
        return [Chunk(rec.id, 0, rec.title, f"{header}\n{rec.text.strip()}")]

    chunks: list[Chunk] = []
    for section_title, body in _split_sections(rec.text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-•] )|\n(?=Q: )", body) if p.strip()]
        for window in _windows(paragraphs, max_words, overlap_words):
            label = f"{section_title}: " if section_title else ""
            chunks.append(Chunk(rec.id, len(chunks), rec.title, f"{header}\n{label}{window}"))
    return chunks or [Chunk(rec.id, 0, rec.title, f"{header}\n{rec.text.strip()[:2000]}")]
