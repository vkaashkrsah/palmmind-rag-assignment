from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import chat, ingestion
from app.db import init_db
from app.deps import get_vectors


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_db()
    await get_vectors().ensure_collection()
    yield


app = FastAPI(title="Palm Mind RAG Backend", version="1.0.0", lifespan=lifespan)
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.exception_handler(httpx.HTTPError)
async def upstream_error(_: Request, exc: httpx.HTTPError) -> JSONResponse:
    return JSONResponse({"detail": f"LLM service error: {exc}"}, status_code=502)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
