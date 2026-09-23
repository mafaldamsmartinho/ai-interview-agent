from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


class ModelProvider(str, Enum):
    QWEN = "qwen"
    LLAMA = "llama"


def get_model(provider: ModelProvider) -> BaseChatModel:
    if provider == ModelProvider.QWEN:
        return ChatOllama(
            model="qwen3:1.7b",
            temperature=0.4,
        )

    if provider == ModelProvider.LLAMA:
        return ChatOllama(
            model="llama3.2:1b",
            temperature=0.4,
        )

    raise ValueError(f"Unsupported model provider: {provider}")
