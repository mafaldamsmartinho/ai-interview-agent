from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


class ModelProvider(str, Enum):
    FAST = "fast"
    STRONG = "strong"


def get_model(provider: ModelProvider) -> BaseChatModel:
    if provider == ModelProvider.STRONG:
        return ChatOllama(
            model="qwen3:1.7b",
            temperature=0.0,
        )

    if provider == ModelProvider.FAST:
        return ChatOllama(
            model="gemma3:1b",
            temperature=0.0,
        )

    raise ValueError(f"Unsupported model provider: {provider}")


def get_router_model() -> BaseChatModel:
    return ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0,
    )
