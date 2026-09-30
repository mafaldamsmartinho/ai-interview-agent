from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


class ModelProvider(str, Enum):
    FAST = "fast"
    STRONG = "strong"


def _create_model(model_name: str) -> BaseChatModel:
    return ChatOllama(
        model=model_name,
        temperature=0.0,
        client_kwargs={
            "timeout": 30.0,
        },
    )


def get_model(provider: ModelProvider) -> BaseChatModel:
    if provider == ModelProvider.STRONG:
        return _create_model("qwen3:1.7b")

    if provider == ModelProvider.FAST:
        return _create_model("gemma3:1b")

    raise ValueError(f"Unsupported model provider: {provider}")


def get_router_model() -> BaseChatModel:
    return _create_model("qwen2.5:1.5b")


def get_tool_guard_model() -> BaseChatModel:
    return _create_model("gemma3:1b")
