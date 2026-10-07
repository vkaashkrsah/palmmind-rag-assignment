# Palm Mind RAG Backend

FastAPI backend with two REST APIs: document ingestion and conversational RAG with interview booking.
No LangChain chains, FAISS or Chroma. The RAG pipeline is custom code.

## Stack
FastAPI, Ollama (`llama3.2` chat, `nomic-embed-text` embeddings), Qdrant (vectors), Redis (chat memory),
SQLite via async SQLAlchemy (document metadata and bookings, swap `DATABASE_URL` for Postgres).

## Run
```bash
ollama pull llama3.2 && ollama pull nomic-embed-text   # Ollama must be running on the host
docker compose up --build
```
Docs at http://localhost:8000/docs

## API
**POST /api/v1/documents** (multipart: `file` .pdf/.txt, `strategy` = `fixed` | `sentence`)
```bash
curl -F file=@sample.pdf -F strategy=sentence http://localhost:8000/api/v1/documents
```
Pipeline: extract text, chunk (fixed-size with overlap, or sentence-aware packing), embed in batches,
upsert to Qdrant, save metadata (filename, strategy, chunk count) in SQL.

**POST /api/v1/chat** (`{"session_id": null, "message": "..."}`), reuse the returned `session_id` for multi-turn
```bash
curl -X POST localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"message":"What does the document say about refunds?"}'
curl -X POST localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"<id>","message":"Book an interview for Ram Sharma, ram@example.com, tomorrow 3pm"}'
```
**GET /api/v1/bookings** lists stored bookings.

## Design
- **Custom RAG** (`services/rag.py`): follow-up questions are rewritten into standalone queries using Redis history,
  then embedded, retrieved from Qdrant (top-k), and answered from context only.
- **Chat memory**: Redis list per session (capped, with TTL). Partial booking details are kept in a Redis hash.
- **Booking**: each turn the LLM extracts intent and fields as JSON. Fields are validated (email format, future ISO date,
  HH:MM) and must appear in the user's own words, which guards against hallucination. Missing fields are asked for
  across turns. Once complete, the booking is saved to SQL.
- Typed throughout (pydantic schemas, annotations), dependency-injected services, modular layout.

## Local dev without Docker
`pip install -r requirements.txt`, run Qdrant and Redis (`docker compose up qdrant redis`), copy `.env.example` to `.env`,
then `uvicorn app.main:app --reload`.
