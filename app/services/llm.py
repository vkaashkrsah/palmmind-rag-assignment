import httpx

from app.config import settings

Message = dict[str, str]


class OllamaClient:
    """Thin async wrapper over the Ollama HTTP API (chat + embeddings)."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=settings.ollama_url, timeout=180)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._http.post(
            "/api/embed", json={"model": settings.embed_model, "input": texts}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def chat(self, messages: list[Message], json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if json_mode:
            payload["format"] = "json"
        resp = await self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])
