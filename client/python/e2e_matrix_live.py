#!/usr/bin/env python3
"""Live end-to-end matrix probe against a running grokcli-2api instance.

Usage:
  export OPENAI_BASE_URL=http://127.0.0.1:3000/v1
  export OPENAI_API_KEY=...
  python e2e_matrix_live.py

Each feature is exercised with a real HTTP call when possible.
Network failures are reported as SKIP/FAIL rather than silently ignored.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok2api_client import (
    Grok2APIClient,
    ChatMessage,
    FunctionTool,
    ToolChoice,
    ResponseFormat,
)


def main() -> int:
    base = os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:3000/v1"
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROK2API_API_KEY") or ""
    model = os.environ.get("GROK2API_MODEL") or "grok-4.5"
    client = Grok2APIClient(base_url=base, api_key=key, default_model=model, timeout=120)

    results = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 0) offline structure always first
    offline = client.validate_feature_matrix_payloads()
    record("offline_matrix", all(offline.values()), json.dumps(offline, sort_keys=True))

    # 1) models / connectivity
    try:
        models = client.list_models()
        record("Chat/models", True, f"models={len(models)}")
    except Exception as e:
        record("Chat/models", False, str(e))
        print("Proxy not reachable; live probes aborted after offline checks.")
        return 0 if all(offline.values()) else 1

    # 2) basic chat + reasoning + top_p + seed + stop + response_id
    try:
        r = client.chat(
            [ChatMessage.system("Reply with exactly: PONG"), ChatMessage.user("ping")],
            reasoning_effort="low",
            top_p=0.9,
            seed=1,
            stop=["<<<END>>>"],
            max_tokens=64,
            temperature=0,
        )
        ok = bool(r.content or r.reasoning or r.finish_reason)
        record(
            "Chat+Reasoning+top_p+seed+stop+response_id",
            ok,
            f"id={r.id!r} response_id={r.response_id!r} finish={r.finish_reason!r} content={r.content[:80]!r}",
        )
    except Exception as e:
        record("Chat+Reasoning+top_p+seed+stop+response_id", False, str(e))

    # 3) function call + tool_choice specific + parallel + max_tool_calls
    tools = [
        FunctionTool(
            name="get_time",
            description="Get current time for a timezone",
            parameters={
                "type": "object",
                "properties": {"tz": {"type": "string"}},
                "required": ["tz"],
            },
        )
    ]

    def executor(name: str, args: dict) -> str:
        if name == "get_time":
            return f"12:00 in {args.get('tz', 'UTC')}"
        return f"unknown tool {name}"

    try:
        r = client.run_tool_loop(
            [ChatMessage.user("What time is it in UTC? Use the get_time tool.")],
            tools=tools,
            executor=executor,
            tool_choice=ToolChoice.function("get_time"),
            parallel_tool_calls=False,
            max_tool_calls=2,
            reasoning_effort="low",
            max_tokens=256,
            max_rounds=3,
            stream=True,
        )
        ok = r.has_tool_calls or bool(r.content)
        record(
            "Function call+tool_choice+max_tool_calls+parallel_tools+Tool result",
            ok,
            f"finish={r.finish_reason!r} tools={[(t.name, t.arguments[:60]) for t in r.tool_calls]} content={r.content[:80]!r}",
        )
    except Exception as e:
        record(
            "Function call+tool_choice+max_tool_calls+parallel_tools+Tool result",
            False,
            str(e),
        )

    # 4) JSON mode
    try:
        r = client.chat_json(
            [ChatMessage.user('Return JSON {"ok": true}')],
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            schema_name="ok_schema",
            reasoning_effort="low",
            max_tokens=64,
        )
        record("JSON mode", bool(r.content), r.content[:120])
    except Exception as e:
        record("JSON mode", False, str(e))

    # 5) web search builtin
    try:
        r = client.chat_with_web_search(
            "What is the latest SpaceX launch date? One short sentence.",
            reasoning_effort="low",
            max_tokens=128,
        )
        record("Web search", bool(r.content or r.finish_reason), r.content[:120])
    except Exception as e:
        record("Web search", False, str(e))

    # 6) code interpreter builtin
    try:
        r = client.chat_with_code_interpreter(
            "Use code to compute 17*19 and report only the number.",
            reasoning_effort="low",
            max_tokens=128,
        )
        record("Code interp", bool(r.content or r.finish_reason), r.content[:120])
    except Exception as e:
        record("Code interp", False, str(e))

    # 7) presence/frequency penalty accepted (may be stripped upstream)
    try:
        body = client.build_chat_body(
            [ChatMessage.user("hi")],
            presence_penalty=0.2,
            frequency_penalty=0.1,
            max_tokens=16,
            reasoning_effort="low",
        )
        # Send raw — success means proxy accepted the request even if fields stripped later.
        payload, headers = client.request("POST", "/chat/completions", body=body)
        record(
            "presence_pen+frequency_pen",
            bool(payload.get("choices") or payload.get("id")),
            f"id={payload.get('id')}",
        )
    except Exception as e:
        record("presence_pen+frequency_pen", False, str(e))

    # 8) Vision (tiny 1x1 png unless VISION_IMAGE provided)
    try:
        img = os.environ.get("VISION_IMAGE_URL")
        if not img:
            # 32x32 solid red PNG (1024px, above upstream min 512px)
            img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAANUlEQVR4nO3QsQ0AMAzDsLT//9yeoCkbeYAN6LzZdZf3x0GSKEmUJEoSJYmSREmiJFGSaMoHo8QBPwYSAhsAAAAASUVORK5CYII="
        r = client.chat_vision("What color is this image? One word.", img, max_tokens=32, reasoning_effort="low")
        record("Vision", bool(r.content or r.finish_reason or r.id), r.content[:120])
    except Exception as e:
        record("Vision", False, str(e))

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Live matrix: {passed}/{total} PASS")
    # Offline structure is required; live connectivity may be absent.
    if not offline or not all(offline.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
