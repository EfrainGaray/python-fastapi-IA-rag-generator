"""Document management API — upload, list, and delete indexed documents."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.models.schemas import DocumentInfo, DocumentListResponse, UploadResponse
from app.store.ingest import SUPPORTED_EXTENSIONS, ingest_file

router = APIRouter(tags=["Documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, req: Request) -> UploadResponse:
    """Upload a document (PDF, DOCX, TXT, XLSX), chunk, embed, and index it."""
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    embedding_model = req.app.state.embedding_model
    es = req.app.state.es

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            chunks_indexed = await ingest_file(tmp_path, filename, es, embedding_model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(
        filename=filename,
        chunks_indexed=chunks_indexed,
        message=f"Successfully indexed {chunks_indexed} chunks from '{filename}'.",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(req: Request) -> DocumentListResponse:
    """List all indexed documents with their chunk counts."""
    from app.config import settings

    es = req.app.state.es
    try:
        result = es.search(
            index=settings.elasticsearch_index,
            body={
                "size": 0,
                "aggs": {
                    "files": {
                        "terms": {"field": "filename", "size": 1000}
                    }
                },
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Elasticsearch query failed: {exc}") from exc

    buckets = result.get("aggregations", {}).get("files", {}).get("buckets", [])
    documents = [
        DocumentInfo(filename=b["key"], chunks=b["doc_count"]) for b in buckets
    ]
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/{filename}", status_code=200)
async def delete_document(filename: str, req: Request) -> dict:
    """Delete all chunks for a given filename from the index."""
    from app.config import settings

    es = req.app.state.es
    try:
        result = es.delete_by_query(
            index=settings.elasticsearch_index,
            body={"query": {"term": {"filename": filename}}},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Delete failed: {exc}") from exc

    deleted = result.get("deleted", 0)
    return {"filename": filename, "deleted": deleted, "message": f"Deleted {deleted} chunks."}
