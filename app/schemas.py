from pydantic import BaseModel, ConfigDict, Field

from app.services.chunking import ChunkStrategy


class DocumentOut(BaseModel):
    id: str
    filename: str
    strategy: ChunkStrategy
    chunks: int


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    filename: str
    score: float
    snippet: str


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    date: str
    time: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = []
    booking: BookingOut | None = None
