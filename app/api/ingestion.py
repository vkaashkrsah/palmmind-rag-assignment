import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Document, get_session
from app.deps import get_llm, get_vectors
from app.schemas import DocumentOut
from app.services.chunking import ChunkStrategy, chunk_text
from app.services.extraction import UnsupportedFileType, extract_text
from app.services.llm import OllamaClient
from app.services.vectorstore import VectorStore

router = APIRouter(prefix="/documents", tags=["ingestion"])
BATCH = 32


@router.post("", response_model=DocumentOut, status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: ChunkStrategy = Form(ChunkStrategy.fixed),
    db: AsyncSession = Depends(get_session),
    llm: OllamaClient = Depends(get_llm),
    vectors: VectorStore = Depends(get_vectors),
) -> DocumentOut:
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "File too large")
    filename = file.filename or "upload"
    try:
        text = extract_text(filename, data)
    except UnsupportedFileType as exc:
        raise HTTPException(415, str(exc)) from exc

    chunks = chunk_text(text, strategy)
    if not chunks:
        raise HTTPException(422, "No extractable text found")

    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), BATCH):
        embeddings.extend(await llm.embed(chunks[i : i + BATCH]))

    doc_id = uuid.uuid4().hex
    await vectors.upsert(doc_id, filename, chunks, embeddings)
    db.add(
        Document(
            id=doc_id,
            filename=filename,
            content_type=file.content_type or "",
            chunk_strategy=strategy.value,
            chunk_count=len(chunks),
            char_count=len(text),
        )
    )
    await db.commit()
    return DocumentOut(id=doc_id, filename=filename, strategy=strategy, chunks=len(chunks))
