"""OpenAI-compatible client for grokcli-2api covering the full feature matrix."""

from .client import Grok2APIClient, FeatureMatrix, ChatResult, ToolCall
from .types import (
    ChatMessage,
    FunctionTool,
    BuiltinTool,
    ResponseFormat,
    ReasoningEffort,
    ToolChoice,
)

__all__ = [
    "Grok2APIClient",
    "FeatureMatrix",
    "ChatResult",
    "ToolCall",
    "ChatMessage",
    "FunctionTool",
    "BuiltinTool",
    "ResponseFormat",
    "ReasoningEffort",
    "ToolChoice",
]

__version__ = "1.0.0"
