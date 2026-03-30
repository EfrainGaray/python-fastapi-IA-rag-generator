"""Document ingestion script.

Reads PDF, DOCX, TXT, and XLSX files from a directory, splits them into
chunks with SentenceSplitter, generates embeddings with SentenceTransformer,
and bulk-indexes the results into Elasticsearch.

Usage:
    python scripts/ingest.py /path/to/docs
"""

import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch, helpers
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger
from sentence_transformers import SentenceTransformer

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402


def _create_index(es: Elasticsearch, index_name: str) -> None:
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "filename": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mappings)
        logger.info(f"Index '{index_name}' created.")
    else:
        logger.info(f"Index '{index_name}' already exists.")


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


_READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".xlsx": _read_excel,
    ".docx": _read_docx,
}


def ingest_documents(directory: Path) -> None:
    es = Elasticsearch(settings.elasticsearch_url)
    _create_index(es, settings.elasticsearch_index)

    model = SentenceTransformer(settings.embedding_model)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    processed_path = directory / ".processed_files.txt"
    processed: set[str] = set()
    if processed_path.exists():
        processed = set(processed_path.read_text().splitlines())

    actions: list[dict] = []
    newly_processed: list[str] = []

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.name in processed:
            logger.info(f"Skipping already processed: {file_path.name}")
            continue

        reader = _READERS.get(file_path.suffix.lower())
        if reader is None:
            logger.warning(f"Unsupported format, skipping: {file_path.name}")
            continue

        logger.info(f"Reading {file_path.name}")
        pages = reader(file_path)

        for text, page_number in pages:
            chunks = splitter.split_text(text)
            logger.debug(
                f"  {file_path.name} p{page_number}: {len(chunks)} chunks"
            )
            for chunk in chunks:
                embedding = model.encode(chunk, convert_to_tensor=False).tolist()
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": settings.elasticsearch_index,
                        "_source": {
                            "content": chunk,
                            "filename": file_path.name,
                            "page_number": page_number,
                            "embedding": embedding,
                        },
                    }
                )

        newly_processed.append(file_path.name)

    if actions:
        logger.info(f"Indexing {len(actions)} chunks to Elasticsearch …")
        helpers.bulk(es, actions)
        logger.info("Indexing complete.")
    else:
        logger.info("No new documents to index.")

    if newly_processed:
        with processed_path.open("a") as fh:
            for name in newly_processed:
                fh.write(f"{name}\n")

    es.close()


if __name__ == "__main__":
    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/docs")
    if not docs_dir.is_dir():
        logger.error(f"Directory not found: {docs_dir}")
        sys.exit(1)
    ingest_documents(docs_dir)
