from functools import lru_cache

from fastapi import Depends

from app.services.llm import OllamaClient
from app.services.memory import ChatMemory
from app.services.rag import RagService
from app.services.vectorstore import VectorStore


@lru_cache
def get_llm() -> OllamaClient:
    return OllamaClient()


@lru_cache
def get_vectors() -> VectorStore:
    return VectorStore()


@lru_cache
def get_memory() -> ChatMemory:
    return ChatMemory()


def get_rag(
    llm: OllamaClient = Depends(get_llm),
    vectors: VectorStore = Depends(get_vectors),
    memory: ChatMemory = Depends(get_memory),
) -> RagService:
    return RagService(llm, vectors, memory)
