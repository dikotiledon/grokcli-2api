#!/usr/bin/env python3
"""Offline structural validation of the full feature matrix client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok2api_client import Grok2APIClient, ChatMessage, FunctionTool, ToolChoice, ResponseFormat


def main() -> int:
    client = Grok2APIClient(base_url="http://example.invalid/v1", api_key="test")
    checks = client.validate_feature_matrix_payloads()
    payloads = client.feature_matrix_payloads()

    print("=== Feature matrix offline validation ===")
    all_ok = True
    for name, ok in checks.items():
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{mark}] {name}")

    # Extra deep checks on function-call body (agent-critical path)
    fc = payloads["function_call"]
    assert fc["tool_choice"]["type"] == "function"
    assert fc["tool_choice"]["function"]["name"] == "lookup"
    assert fc["parallel_tool_calls"] is True
    assert fc["max_tool_calls"] == 3
    assert any(t.get("type") == "function" for t in fc["tools"])

    tr = payloads["tool_result"]
    tool_msg = next(m for m in tr["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "12:00 UTC"

    vision = payloads["vision"]
    parts = vision["messages"][0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)

    ws = payloads["web_search"]
    assert any(t.get("type") == "web_search" for t in ws["tools"])

    ci = payloads["code_interpreter"]
    assert any(t.get("type") == "code_interpreter" for t in ci["tools"])

    js = payloads["json_mode_schema"]
    assert js["response_format"]["type"] == "json_schema"

    sampling = payloads["sampling"]
    assert sampling["stop"] == ["END", "STOP"]
    assert sampling["seed"] == 42
    assert sampling["top_p"] == 0.8

    # Tool loop shape (no network): ensure assistant+tool messages compose
    tools = [FunctionTool(name="lookup", parameters={"type": "object", "properties": {"key": {"type": "string"}}})]
    body = client.build_chat_body(
        [
            ChatMessage.user("x"),
            ChatMessage.assistant(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"key":"a"}'},
                    }
                ]
            ),
            ChatMessage.tool_result("c1", "A"),
        ],
        tools=tools,
        tool_choice=ToolChoice.auto(),
    )
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "tool"]

    print()
    print("sample function_call body:")
    print(json.dumps(fc, indent=2)[:800])
    print()
    if all_ok:
        print("ALL MATRIX FEATURES: PASS (offline structure)")
        return 0
    print("SOME MATRIX FEATURES: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
