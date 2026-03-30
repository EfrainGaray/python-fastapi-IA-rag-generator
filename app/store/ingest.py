"""Shared document ingestion logic used by both the CLI script and the upload API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from elasticsearch import helpers
from loguru import logger

from app.config import settings


def _read_txt(file_path: Path) -> list[tuple[str, int]]:
    return [(file_path.read_text(encoding="utf-8"), 1)]


def _read_pdf(file_path: Path) -> list[tuple[str, int]]:
    from pypdf import PdfReader

    pages = []
    reader = PdfReader(str(file_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, page_num))
    return pages


def _read_excel(file_path: Path) -> list[tuple[str, int]]:
    import pandas as pd

    df = pd.read_excel(file_path)
    return [(df.to_string(index=False), 1)]


def _read_docx(file_path: Path) -> list[tuple[str, int]]:
    import docx

    doc = docx.Document(str(file_path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(text, 1)]


_READERS: dict[str, Any] = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".xlsx": _read_excel,
    ".docx": _read_docx,
}

SUPPORTED_EXTENSIONS: set[str] = set(_READERS.keys())


def _chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 20) -> list[str]:
    """Simple word-based chunking that respects chunk_size (approx chars) with overlap."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words) and length + len(words[end]) + 1 <= chunk_size:
            length += len(words[end]) + 1
            end += 1
        if end == start:
            # Single word longer than chunk_size — take it alone
            end = start + 1
        chunks.append(" ".join(words[start:end]))
        # Overlap: step back by overlap characters worth of words
        overlap_chars = 0
        step_back = 0
        for i in range(end - 1, start - 1, -1):
            overlap_chars += len(words[i]) + 1
            step_back += 1
            if overlap_chars >= chunk_overlap:
                break
        start = end - step_back if end - step_back > start else end
    return chunks


async def ingest_file(
    file_path: Path,
    filename: str,
    es: Any,
    embedding_model: Any,
) -> int:
    """Read file, chunk, embed, bulk-index to ES. Returns number of chunks indexed."""
    suffix = file_path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(f"Unsupported file format: {suffix!r}")

    logger.info(f"Ingesting file: {filename}")
    pages = reader(file_path)

    loop = asyncio.get_event_loop()

    actions: list[dict] = []
    for text, page_number in pages:
        chunks = _chunk_text(text, chunk_size=512, chunk_overlap=20)
        logger.debug(f"  {filename} p{page_number}: {len(chunks)} chunks")
        for chunk in chunks:
            raw_embedding = await loop.run_in_executor(
                None,
                lambda c=chunk: embedding_model.encode(c, convert_to_tensor=False),
            )
            embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": settings.elasticsearch_index,
                    "_source": {
                        "content": chunk,
                        "filename": filename,
                        "page_number": page_number,
                        "embedding": embedding,
                    },
                }
            )

    if actions:
        helpers.bulk(es, actions)
        logger.info(f"Indexed {len(actions)} chunks for {filename}")

    return len(actions)
