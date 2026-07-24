"""Typed request builders for the grokcli-2api feature matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

ReasoningEffort = Literal["minimal", "low", "medium", "high", "auto"]


def _text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _image_part(url: str, detail: Optional[str] = None) -> Dict[str, Any]:
    image: Dict[str, Any] = {"url": url}
    if detail:
        image["detail"] = detail
    return {"type": "image_url", "image_url": image}


@dataclass
class ChatMessage:
    """OpenAI chat message with vision + tool-result support."""

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None

    @classmethod
    def system(cls, text: str) -> "ChatMessage":
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str) -> "ChatMessage":
        return cls(role="user", content=text)

    @classmethod
    def user_with_image(
        cls,
        text: str,
        image_url: str,
        *,
        detail: Optional[str] = None,
    ) -> "ChatMessage":
        """Vision: multimodal user message (image base64 data URL or http URL)."""
        return cls(
            role="user",
            content=[_text_part(text), _image_part(image_url, detail=detail)],
        )

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
    ) -> "ChatMessage":
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        *,
        name: Optional[str] = None,
    ) -> "ChatMessage":
        """Multi-turn tool conversation: role=tool output for a prior tool_call."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        elif self.tool_calls:
            out["content"] = None
        else:
            out["content"] = ""
        if self.name:
            out["name"] = self.name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.reasoning_content:
            out["reasoning_content"] = self.reasoning_content
        return out


@dataclass
class FunctionTool:
    """Client-side function tool (OpenAI chat tools[] entry)."""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        fn: Dict[str, Any] = {
            "name": self.name,
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }
        if self.description:
            fn["description"] = self.description
        if self.strict is not None:
            fn["strict"] = self.strict
        return {"type": "function", "function": fn}


@dataclass
class BuiltinTool:
    """Server-side builtin tool: web_search / code_interpreter / x_search."""

    type: Literal[
        "web_search",
        "web_search_preview",
        "code_interpreter",
        "code_execution",
        "x_search",
        "live_search",
    ]
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {"type": self.type}
        out.update(self.options)
        return out


@dataclass
class ResponseFormat:
    """JSON mode: json_object or json_schema."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    strict: Optional[bool] = None

    @classmethod
    def json_object(cls) -> "ResponseFormat":
        return cls(type="json_object")

    @classmethod
    def json_schema_obj(
        cls,
        schema: Mapping[str, Any],
        *,
        name: str = "response",
        strict: bool = True,
    ) -> "ResponseFormat":
        return cls(
            type="json_schema",
            name=name,
            strict=strict,
            json_schema={
                "name": name,
                "strict": strict,
                "schema": dict(schema),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.type == "json_schema":
            out: Dict[str, Any] = {"type": "json_schema"}
            if self.json_schema is not None:
                out["json_schema"] = self.json_schema
            if self.name:
                out["name"] = self.name
            return out
        return {"type": self.type}


class ToolChoice:
    """tool_choice helpers: auto / none / required / specific function."""

    @staticmethod
    def auto() -> str:
        return "auto"

    @staticmethod
    def none() -> str:
        return "none"

    @staticmethod
    def required() -> str:
        return "required"

    @staticmethod
    def function(name: str) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": name}}

    @staticmethod
    def builtin(type_name: str) -> Dict[str, Any]:
        return {"type": type_name}


def encode_image_base64(path: Union[str, Path], *, mime: Optional[str] = None) -> str:
    """Load a local image as a data URL (min 512px recommended by vision matrix)."""
    import base64
    import mimetypes

    p = Path(path)
    data = p.read_bytes()
    if mime is None:
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def ensure_min_image_side(path: Union[str, Path], min_side: int = 512) -> Path:
    """Optionally upscale image so shortest side >= min_side (vision matrix).

    Requires Pillow. Returns path to a temp PNG if resized, else original path.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for image resize: pip install pillow"
        ) from exc

    p = Path(path)
    img = Image.open(p)
    w, h = img.size
    short = min(w, h)
    if short >= min_side:
        return p
    scale = min_side / float(short)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    out = p.with_name(f"{p.stem}_min{min_side}.png")
    resized.save(out, format="PNG")
    return out
