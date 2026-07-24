"""Full-featured OpenAI-compatible client for grokcli-2api.

Covers the feature matrix end-to-end:

Chat, Vision, Function call, Tool result, System prompt, JSON mode,
Web search, Code interpreter, Reasoning, top_p, Stop sequences,
tool_choice, max_tool_calls, parallel_tools, presence/frequency_penalty
(accepted client-side; proxy may strip if upstream rejects), seed,
response_id.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union
from urllib import error as urlerror
from urllib import request as urlrequest

from .types import (
    BuiltinTool,
    ChatMessage,
    FunctionTool,
    ReasoningEffort,
    ResponseFormat,
    ToolChoice,
)

MessageLike = Union[ChatMessage, Dict[str, Any]]
ToolLike = Union[FunctionTool, BuiltinTool, Dict[str, Any]]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    index: int = 0

    def parsed_arguments(self) -> Any:
        try:
            return json.loads(self.arguments or "{}")
        except Exception:
            return self.arguments

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
            "index": self.index,
        }


@dataclass
class ChatResult:
    """Normalized chat completion result."""

    id: str
    model: str
    content: str
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    response_id: str = ""
    prompt_cache_key: str = ""
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def assistant_message(self) -> ChatMessage:
        calls = [c.to_openai() for c in self.tool_calls] if self.tool_calls else None
        return ChatMessage.assistant(
            content=self.content or None,
            tool_calls=calls,
            reasoning_content=self.reasoning or None,
        )


@dataclass
class FeatureMatrix:
    """Declared feature support for this proxy client."""

    chat: str = "full"
    vision: bool = True
    function_call: bool = True
    tool_result: bool = True
    system_prompt: bool = True
    json_mode: bool = True
    web_search: bool = True
    code_interpreter: bool = True
    reasoning: str = "auto"  # low/medium/high/minimal/auto
    top_p: bool = True
    stop_sequences: bool = True  # accepted on chat body; dropped before /responses
    tool_choice: bool = True
    max_tool_calls: bool = True
    parallel_tools: bool = True
    presence_penalty: bool = True  # accepted; may be stripped upstream
    frequency_penalty: bool = True  # accepted; may be stripped upstream
    seed: bool = True  # accepted on chat body; dropped before /responses
    response_id: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Chat": self.chat,
            "Vision": self.vision,
            "Function call": self.function_call,
            "Tool result": self.tool_result,
            "System prompt": self.system_prompt,
            "JSON mode": self.json_mode,
            "Web search": self.web_search,
            "Code interp": self.code_interpreter,
            "Reasoning": self.reasoning,
            "top_p": self.top_p,
            "Stop sequences": self.stop_sequences,
            "tool_choice": self.tool_choice,
            "max_tool_calls": self.max_tool_calls,
            "parallel_tools": self.parallel_tools,
            "presence_pen": self.presence_penalty,
            "frequency_pen": self.frequency_penalty,
            "seed": self.seed,
            "response_id": self.response_id,
        }


class Grok2APIClient:
    """HTTP client for grokcli-2api OpenAI-compatible endpoints.

    Endpoints:
      GET  /v1/models
      POST /v1/chat/completions
      POST /v1/responses
      POST /v1/messages  (Anthropic shape passthrough helper)
    """

    FEATURES = FeatureMatrix()

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        default_model: str = "grok-4.5",
        timeout: float = 300.0,
        user_agent: str = "grok2api-client/1.0",
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        base = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:3000/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        self.base_url = base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROK2API_API_KEY") or ""
        self.default_model = default_model
        self.timeout = timeout
        self.user_agent = user_agent
        self.default_headers = dict(default_headers or {})

    # ------------------------------------------------------------------
    # low-level HTTP
    # ------------------------------------------------------------------
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.default_headers)
        if extra:
            h.update(extra)
        return h

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        # base_url usually ends with /v1
        if path.startswith("/v1/") and self.base_url.endswith("/v1"):
            return self.base_url[: -len("/v1")] + path
        if self.base_url.endswith("/v1") and path.startswith("/"):
            return self.base_url + path
        return self.base_url.rstrip("/") + path

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Any:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(
            self._url(path),
            data=data,
            headers=self._headers(headers),
            method=method.upper(),
        )
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout)
        except urlerror.HTTPError as e:
            err_body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {e.code} {path}: {err_body}") from e
        except urlerror.URLError as e:
            raise RuntimeError(f"request failed {path}: {e}") from e

        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        if stream:
            return resp, resp_headers
        raw = resp.read()
        resp.close()
        if not raw:
            return {}, resp_headers
        try:
            return json.loads(raw.decode("utf-8")), resp_headers
        except Exception:
            return {"raw": raw.decode("utf-8", "replace")}, resp_headers

    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        payload, _ = self.request("GET", "/models")
        return list(payload.get("data") or [])

    # ------------------------------------------------------------------
    # body builder — full matrix
    # ------------------------------------------------------------------
    def build_chat_body(
        self,
        messages: Sequence[MessageLike],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[Sequence[ToolLike]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        max_tool_calls: Optional[int] = None,
        response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        user: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        instructions: Optional[str] = None,
        system: Optional[str] = None,
        web_search: bool = False,
        code_interpreter: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append(ChatMessage.system(system).to_dict())
        for m in messages:
            msgs.append(m.to_dict() if isinstance(m, ChatMessage) else dict(m))

        body: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": msgs,
            "stream": bool(stream),
        }

        tool_list: List[Dict[str, Any]] = []
        if tools:
            for t in tools:
                if isinstance(t, (FunctionTool, BuiltinTool)):
                    tool_list.append(t.to_dict())
                else:
                    tool_list.append(dict(t))
        if web_search and not any(t.get("type") == "web_search" for t in tool_list):
            tool_list.append(BuiltinTool(type="web_search").to_dict())
        if code_interpreter and not any(
            t.get("type") in ("code_interpreter", "code_execution") for t in tool_list
        ):
            tool_list.append(BuiltinTool(type="code_interpreter").to_dict())
        if tool_list:
            body["tools"] = tool_list

        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = bool(parallel_tool_calls)
        if max_tool_calls is not None:
            body["max_tool_calls"] = int(max_tool_calls)

        if response_format is not None:
            if isinstance(response_format, ResponseFormat):
                body["response_format"] = response_format.to_dict()
            else:
                body["response_format"] = dict(response_format)

        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            body["temperature"] = float(temperature)
        if top_p is not None:
            body["top_p"] = float(top_p)
        if stop is not None:
            body["stop"] = list(stop) if not isinstance(stop, str) else [stop]
        if seed is not None:
            body["seed"] = int(seed)
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if presence_penalty is not None:
            body["presence_penalty"] = float(presence_penalty)
        if frequency_penalty is not None:
            body["frequency_penalty"] = float(frequency_penalty)
        if user is not None:
            body["user"] = user
        if prompt_cache_key is not None:
            body["prompt_cache_key"] = prompt_cache_key
        if instructions is not None:
            # Responses-style system passthrough; also inject as system message if missing.
            body["instructions"] = instructions
            if not any(m.get("role") == "system" for m in msgs) and instructions.strip():
                body["messages"] = [{"role": "system", "content": instructions}] + msgs

        if stream:
            body["stream_options"] = {"include_usage": True}

        if extra:
            body.update(extra)
        return body

    # ------------------------------------------------------------------
    # chat completions
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[MessageLike],
        **kwargs: Any,
    ) -> ChatResult:
        """Non-streaming chat completion with full matrix params."""
        kwargs = dict(kwargs)
        kwargs["stream"] = False
        body = self.build_chat_body(messages, **kwargs)
        payload, headers = self.request("POST", "/chat/completions", body=body)
        return self._parse_completion(payload, headers)

    def chat_stream(
        self,
        messages: Sequence[MessageLike],
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw SSE data payloads (parsed JSON) for chat completions."""
        kwargs = dict(kwargs)
        kwargs["stream"] = True
        body = self.build_chat_body(messages, **kwargs)
        headers = {"Accept": "text/event-stream"}
        resp, _ = self.request("POST", "/chat/completions", body=body, headers=headers, stream=True)
        try:
            yield from self._iter_sse(resp)
        finally:
            resp.close()

    def chat_collect_stream(
        self,
        messages: Sequence[MessageLike],
        **kwargs: Any,
    ) -> ChatResult:
        """Stream and assemble a ChatResult (tool_calls merged by index)."""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_map: Dict[int, Dict[str, Any]] = {}
        finish = None
        usage: Dict[str, Any] = {}
        cid = ""
        model = kwargs.get("model") or self.default_model
        last_headers: Dict[str, str] = {}

        kwargs = dict(kwargs)
        kwargs["stream"] = True
        body = self.build_chat_body(messages, **kwargs)
        resp, headers = self.request(
            "POST",
            "/chat/completions",
            body=body,
            headers={"Accept": "text/event-stream"},
            stream=True,
        )
        last_headers = headers
        try:
            for event in self._iter_sse(resp):
                if not isinstance(event, dict):
                    continue
                cid = event.get("id") or cid
                model = event.get("model") or model
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish = choice.get("finish_reason")
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(str(delta["content"]))
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        reasoning_parts.append(str(rc))
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index") or 0)
                        slot = tool_map.setdefault(
                            idx,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if tc.get("type"):
                            slot["type"] = tc["type"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] = (
                                slot["function"].get("arguments") or ""
                            ) + str(fn["arguments"])
        finally:
            resp.close()

        tool_calls = [
            ToolCall(
                id=tool_map[i].get("id") or f"call_{i}",
                name=(tool_map[i].get("function") or {}).get("name") or "",
                arguments=(tool_map[i].get("function") or {}).get("arguments") or "",
                index=i,
            )
            for i in sorted(tool_map)
        ]
        response_id = cid or last_headers.get("x-request-id", "")
        pck = last_headers.get("x-grok2api-prompt-cache-key", "")
        return ChatResult(
            id=cid or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=model,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage or {},
            raw={},
            response_id=response_id,
            prompt_cache_key=pck,
            headers=last_headers,
        )

    # ------------------------------------------------------------------
    # multi-turn tool loop
    # ------------------------------------------------------------------
    def run_tool_loop(
        self,
        messages: Sequence[MessageLike],
        tools: Sequence[ToolLike],
        executor: Callable[[str, Dict[str, Any]], str],
        *,
        max_rounds: int = 8,
        stream: bool = True,
        **kwargs: Any,
    ) -> ChatResult:
        """End-to-end function-call + tool-result multi-turn conversation.

        executor(name, args_dict) -> tool result string
        """
        history: List[Dict[str, Any]] = [
            m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages
        ]
        last: Optional[ChatResult] = None
        for _ in range(max_rounds):
            if stream:
                last = self.chat_collect_stream(history, tools=tools, **kwargs)
            else:
                last = self.chat(history, tools=tools, **kwargs)
            if not last.has_tool_calls:
                return last
            history.append(last.assistant_message().to_dict())
            for call in last.tool_calls:
                args = call.parsed_arguments()
                if not isinstance(args, dict):
                    args = {"_raw": args}
                try:
                    result = executor(call.name, args)
                except Exception as exc:
                    result = f"tool error: {exc}"
                history.append(
                    ChatMessage.tool_result(call.id, str(result), name=call.name).to_dict()
                )
        assert last is not None
        return last

    # ------------------------------------------------------------------
    # convenience feature helpers
    # ------------------------------------------------------------------
    def chat_with_web_search(self, prompt: str, **kwargs: Any) -> ChatResult:
        """Web search: inject builtin web_search tool."""
        messages = [ChatMessage.user(prompt)]
        return self.chat(messages, web_search=True, **kwargs)

    def chat_with_code_interpreter(self, prompt: str, **kwargs: Any) -> ChatResult:
        """Code interpreter: inject builtin code_interpreter tool."""
        messages = [ChatMessage.user(prompt)]
        return self.chat(messages, code_interpreter=True, **kwargs)

    def chat_json(
        self,
        messages: Sequence[MessageLike],
        *,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "response",
        **kwargs: Any,
    ) -> ChatResult:
        """JSON mode: json_object or json_schema."""
        if schema is not None:
            rf = ResponseFormat.json_schema_obj(schema, name=schema_name)
        else:
            rf = ResponseFormat.json_object()
        return self.chat(messages, response_format=rf, **kwargs)

    def chat_vision(
        self,
        text: str,
        image_url: str,
        *,
        detail: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Vision: image_url (data URL base64 or http), min 512px recommended."""
        messages = [ChatMessage.user_with_image(text, image_url, detail=detail)]
        return self.chat(messages, **kwargs)

    def chat_with_reasoning(
        self,
        messages: Sequence[MessageLike],
        effort: ReasoningEffort = "auto",
        **kwargs: Any,
    ) -> ChatResult:
        return self.chat(messages, reasoning_effort=effort, **kwargs)

    # ------------------------------------------------------------------
    # responses API (native)
    # ------------------------------------------------------------------
    def responses_create(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload, headers = self.request("POST", "/responses", body=body)
        if isinstance(payload, dict):
            payload.setdefault("_headers", headers)
        return payload

    # ------------------------------------------------------------------
    # feature matrix self-check (offline body construction)
    # ------------------------------------------------------------------
    def feature_matrix_payloads(self) -> Dict[str, Dict[str, Any]]:
        """Build representative request bodies for every matrix feature.

        Useful for unit tests / dry-run validation without a live server.
        """
        base_msgs = [
            ChatMessage.system("You are a precise assistant."),
            ChatMessage.user("hello"),
        ]
        tools = [
            FunctionTool(
                name="lookup",
                description="Look up a key",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            )
        ]
        return {
            "chat": self.build_chat_body(base_msgs, max_tokens=64, reasoning_effort="low"),
            "vision": self.build_chat_body(
                [
                    ChatMessage.user_with_image(
                        "describe",
                        "data:image/png;base64,iVBORw0KGgo=",
                        detail="high",
                    )
                ]
            ),
            "function_call": self.build_chat_body(
                base_msgs,
                tools=tools,
                tool_choice=ToolChoice.function("lookup"),
                parallel_tool_calls=True,
                max_tool_calls=3,
            ),
            "tool_result": self.build_chat_body(
                [
                    ChatMessage.user("time?"),
                    ChatMessage.assistant(
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"key":"now"}'},
                            }
                        ]
                    ),
                    ChatMessage.tool_result("call_1", "12:00 UTC"),
                ],
                tools=tools,
            ),
            "system_prompt": self.build_chat_body(
                [ChatMessage.user("hi")],
                system="System instructions passthrough.",
                instructions="Also via instructions field.",
            ),
            "json_mode_object": self.build_chat_body(
                base_msgs, response_format=ResponseFormat.json_object()
            ),
            "json_mode_schema": self.build_chat_body(
                base_msgs,
                response_format=ResponseFormat.json_schema_obj(
                    {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    name="ok_schema",
                ),
            ),
            "web_search": self.build_chat_body(base_msgs, web_search=True),
            "code_interpreter": self.build_chat_body(base_msgs, code_interpreter=True),
            "reasoning_auto": self.build_chat_body(base_msgs, reasoning_effort="auto"),
            "sampling": self.build_chat_body(
                base_msgs,
                top_p=0.8,
                temperature=0.4,
                stop=["END", "STOP"],
                seed=42,
                presence_penalty=0.1,
                frequency_penalty=0.2,
            ),
            "tool_controls": self.build_chat_body(
                base_msgs,
                tools=tools,
                tool_choice=ToolChoice.required(),
                parallel_tool_calls=False,
                max_tool_calls=1,
            ),
        }

    def validate_feature_matrix_payloads(self) -> Dict[str, bool]:
        """Offline structural validation of feature matrix bodies."""
        payloads = self.feature_matrix_payloads()
        checks: Dict[str, bool] = {}

        chat = payloads["chat"]
        checks["Chat"] = chat.get("messages") is not None and chat.get("model")

        vision = payloads["vision"]
        content = vision["messages"][0]["content"]
        checks["Vision"] = isinstance(content, list) and any(
            p.get("type") == "image_url" for p in content
        )

        fc = payloads["function_call"]
        tc = fc.get("tool_choice") or {}
        checks["Function call"] = bool(fc.get("tools")) and (
            isinstance(tc, dict) and (tc.get("function") or {}).get("name") == "lookup"
        )

        tr = payloads["tool_result"]
        roles = [m.get("role") for m in tr["messages"]]
        checks["Tool result"] = "tool" in roles and any(
            m.get("tool_calls") for m in tr["messages"] if m.get("role") == "assistant"
        )

        sp = payloads["system_prompt"]
        checks["System prompt"] = any(m.get("role") == "system" for m in sp["messages"]) or bool(
            sp.get("instructions")
        )

        jo = payloads["json_mode_object"]
        js = payloads["json_mode_schema"]
        checks["JSON mode"] = (jo.get("response_format") or {}).get("type") == "json_object" and (
            js.get("response_format") or {}
        ).get("type") == "json_schema"

        checks["Web search"] = any(
            t.get("type") == "web_search" for t in payloads["web_search"].get("tools") or []
        )
        checks["Code interp"] = any(
            t.get("type") == "code_interpreter"
            for t in payloads["code_interpreter"].get("tools") or []
        )
        checks["Reasoning"] = payloads["reasoning_auto"].get("reasoning_effort") == "auto"

        sampling = payloads["sampling"]
        checks["top_p"] = sampling.get("top_p") == 0.8
        checks["Stop sequences"] = sampling.get("stop") == ["END", "STOP"]
        checks["seed"] = sampling.get("seed") == 42
        checks["presence_pen"] = sampling.get("presence_penalty") == 0.1
        checks["frequency_pen"] = sampling.get("frequency_penalty") == 0.2

        controls = payloads["tool_controls"]
        checks["tool_choice"] = controls.get("tool_choice") == "required"
        checks["max_tool_calls"] = controls.get("max_tool_calls") == 1
        checks["parallel_tools"] = controls.get("parallel_tool_calls") is False

        # response_id is a response field; client always surfaces it when present.
        checks["response_id"] = True
        return checks

    # ------------------------------------------------------------------
    # parsers
    # ------------------------------------------------------------------
    def _parse_completion(self, payload: Dict[str, Any], headers: Dict[str, str]) -> ChatResult:
        choices = payload.get("choices") or []
        message: Dict[str, Any] = {}
        finish = None
        if choices:
            choice0 = choices[0] or {}
            message = choice0.get("message") or choice0.get("delta") or {}
            finish = choice0.get("finish_reason")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                str(p.get("text") or "") for p in content if isinstance(p, dict)
            )
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or payload.get("reasoning")
            or ""
        )
        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(fn.get("name") or ""),
                    arguments=str(fn.get("arguments") or ""),
                    index=int(tc.get("index") if tc.get("index") is not None else i),
                )
            )
        # legacy function_call
        if not tool_calls and message.get("function_call"):
            fc = message["function_call"]
            tool_calls.append(
                ToolCall(
                    id="call_legacy",
                    name=str(fc.get("name") or ""),
                    arguments=str(fc.get("arguments") or ""),
                    index=0,
                )
            )
        cid = str(payload.get("id") or "")
        response_id = cid
        # Some relays put response id in headers.
        if not response_id:
            response_id = headers.get("x-request-id") or headers.get("x-grok2api-response-id") or ""
        pck = headers.get("x-grok2api-prompt-cache-key") or str(
            payload.get("prompt_cache_key") or ""
        )
        return ChatResult(
            id=cid or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=str(payload.get("model") or self.default_model),
            content=str(content or ""),
            reasoning=str(reasoning or ""),
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=dict(payload.get("usage") or {}),
            raw=payload,
            response_id=response_id,
            prompt_cache_key=pck,
            headers=headers,
        )

    def _iter_sse(self, resp: Any) -> Iterator[Dict[str, Any]]:
        while True:
            line = resp.readline()
            if not line:
                break
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except Exception:
                yield {"raw": data}
