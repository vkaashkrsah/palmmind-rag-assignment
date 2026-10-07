import datetime as dt
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Booking
from app.schemas import BookingOut, ChatResponse, Source
from app.services.llm import Message, OllamaClient
from app.services.memory import ChatMemory
from app.services.vectorstore import VectorStore

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
REQUIRED = ("name", "email", "date", "time")
LABELS = {
    "name": "full name",
    "email": "email address",
    "date": "preferred date",
    "time": "preferred time",
}
ANSWER_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the context below. "
    "If the answer is not in the context, say you don't know. Be concise.\n\n"
    "Context:\n{context}"
)


class Extraction(BaseModel):
    intent: Literal["booking", "question"] = "question"
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class RagService:
    """Custom RAG pipeline: condense -> embed -> retrieve -> generate, plus LLM booking flow."""

    def __init__(self, llm: OllamaClient, vectors: VectorStore, memory: ChatMemory) -> None:
        self.llm, self.vectors, self.memory = llm, vectors, memory

    async def chat(self, db: AsyncSession, session_id: str | None, message: str) -> ChatResponse:
        sid = session_id or uuid.uuid4().hex
        history = await self.memory.history(sid)
        state = await self.memory.get_booking(sid)

        extraction = await self._extract(history, message)
        user_text = " ".join([m["content"] for m in history if m["role"] == "user"] + [message])
        fields = self._validate(extraction, user_text)

        sources: list[Source] = []
        booking: BookingOut | None = None
        if extraction.intent == "booking" or (state and fields):
            state.update(fields)
            missing = [LABELS[k] for k in REQUIRED if k not in state]
            if missing:
                await self.memory.set_booking(sid, state)
                answer = f"Happy to book your interview. Please share your {', '.join(missing)}."
            else:
                row = Booking(session_id=sid, **{k: state[k] for k in REQUIRED})
                db.add(row)
                await db.commit()
                await self.memory.clear_booking(sid)
                booking = BookingOut.model_validate(row)
                answer = (
                    f"Done! Interview booked for {row.name} on {row.date} at {row.time}. "
                    f"A confirmation will go to {row.email}."
                )
        else:
            answer, sources = await self._rag_answer(history, message)

        await self.memory.add(sid, "user", message)
        await self.memory.add(sid, "assistant", answer)
        return ChatResponse(session_id=sid, answer=answer, sources=sources, booking=booking)

    async def _rag_answer(self, history: list[Message], message: str) -> tuple[str, list[Source]]:
        query = await self._condense(history, message)
        vector = (await self.llm.embed([query]))[0]
        hits = await self.vectors.search(vector, settings.top_k)
        context = "\n\n".join(f"[{i}] ({h.filename}) {h.text}" for i, h in enumerate(hits, 1))
        messages: list[Message] = [
            {"role": "system", "content": ANSWER_PROMPT.format(context=context or "(none)")},
            *history,
            {"role": "user", "content": message},
        ]
        answer = await self.llm.chat(messages)
        sources = [Source(filename=h.filename, score=h.score, snippet=h.text[:200]) for h in hits]
        return answer, sources

    async def _condense(self, history: list[Message], message: str) -> str:
        """Rewrite a follow-up into a standalone question (multi-turn handling)."""
        if not history:
            return message
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        out = await self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "Rewrite the user's last message as a standalone question using "
                    "the conversation for context. Return only the question.",
                },
                {"role": "user", "content": f"{transcript}\nuser: {message}"},
            ]
        )
        return out.strip() or message

    async def _extract(self, history: list[Message], message: str) -> Extraction:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        system = (
            f"Today is {dt.date.today().isoformat()}. Return JSON only: "
            '{"intent":"booking"|"question","name":string|null,"email":string|null,'
            '"date":"YYYY-MM-DD"|null,"time":"HH:MM" (24h)|null}. '
            'intent is "booking" if the user wants to schedule/book an interview or is '
            'providing booking details, otherwise "question". Fill a field only if the user '
            "explicitly stated it, else null. Resolve relative dates like 'tomorrow'."
        )
        raw = await self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest: {message}"},
            ],
            json_mode=True,
        )
        try:
            return Extraction.model_validate_json(raw)
        except ValidationError:
            return Extraction()

    @staticmethod
    def _validate(ex: Extraction, user_text: str) -> dict[str, str]:
        """Keep only well-formed fields that really appear in what the user said."""
        out: dict[str, str] = {}
        clean = lambda v: None if not v or v.strip().lower() in {"null", "none", "n/a"} else v.strip()  # noqa: E731
        date_hint = re.compile(
            r"\d{1,2}[/-]\d{1,2}|\b(today|tomorrow|next|mon|tue|wed|thu|fri|sat|sun"
            r"|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            re.I,
        )
        time_hint = re.compile(r"\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)\b|\bnoon\b", re.I)
        name, email = clean(ex.name), clean(ex.email)
        if name and name.lower() in user_text.lower():
            out["name"] = name
        if email and EMAIL_RE.fullmatch(email) and email.lower() in user_text.lower():
            out["email"] = email
        try:
            if (d := clean(ex.date)) and date_hint.search(user_text):
                if dt.date.fromisoformat(d) >= dt.date.today():
                    out["date"] = d
            if (t := clean(ex.time)) and time_hint.search(user_text):
                out["time"] = dt.time.fromisoformat(t).strftime("%H:%M")
        except ValueError:
            pass
        return out
