import uuid

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings


class Hit(BaseModel):
    text: str
    filename: str
    score: float


class VectorStore:
    """Qdrant-backed store for document chunks."""

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
            )

    async def upsert(
        self, doc_id: str, filename: str, chunks: list[str], vectors: list[list[float]]
    ) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"doc_id": doc_id, "filename": filename, "chunk_index": i, "text": text},
            )
            for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await self._client.upsert(self._collection, points=points)

    async def search(self, vector: list[float], limit: int) -> list[Hit]:
        res = await self._client.query_points(
            self._collection, query=vector, limit=limit, with_payload=True
        )
        return [
            Hit(
                text=str((p.payload or {}).get("text", "")),
                filename=str((p.payload or {}).get("filename", "")),
                score=p.score,
            )
            for p in res.points
        ]
