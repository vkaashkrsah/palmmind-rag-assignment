from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Booking, get_session
from app.deps import get_rag
from app.schemas import BookingOut, ChatRequest, ChatResponse
from app.services.rag import RagService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_session),
    rag: RagService = Depends(get_rag),
) -> ChatResponse:
    return await rag.chat(db, body.session_id, body.message)


@router.get("/bookings", response_model=list[BookingOut])
async def list_bookings(db: AsyncSession = Depends(get_session)) -> list[BookingOut]:
    rows = (await db.scalars(select(Booking).order_by(Booking.id))).all()
    return [BookingOut.model_validate(r) for r in rows]
