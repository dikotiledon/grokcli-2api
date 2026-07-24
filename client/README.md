# grokcli-2api client (full feature matrix)

OpenAI-compatible Python client that exercises **every** feature from the
proxy capability matrix end-to-end:

| Fitur | Client support | Notes |
|---|---|---|
| Chat | full | `/v1/chat/completions` (+ stream) |
| Vision | yes | `image_url` / base64 data URL (min 512px recommended) |
| Function call | yes | OpenAI tools → Grok responses bridge |
| Tool result | yes | multi-turn `role=tool` + `run_tool_loop` |
| System prompt | yes | `system` message + `instructions` passthrough |
| JSON mode | yes | `json_object` + `json_schema` → proxy `text.format` |
| Web search | yes | builtin `web_search` tool |
| Code interp | yes | builtin `code_interpreter` tool |
| Reasoning | auto | `minimal/low/medium/high/auto` |
| top_p | yes | pass-through |
| Stop sequences | yes | pass-through |
| tool_choice | yes | auto/none/required/**specific function** |
| max_tool_calls | yes | pass-through |
| parallel_tools | yes | `parallel_tool_calls` |
| presence_pen | accepted | sent by client; proxy may strip for upstream |
| frequency_pen | accepted | sent by client; proxy may strip for upstream |
| seed | yes | pass-through |
| response_id | yes | surfaced from body/headers |

## Install

```bash
cd client/python
# optional editable install
pip install -e .
# vision resize helper
pip install -e ".[vision]"
```

Or just set `PYTHONPATH=client/python`.

## Quick start

```python
from grok2api_client import Grok2APIClient, ChatMessage, FunctionTool, ToolChoice

client = Grok2APIClient(
    base_url="http://127.0.0.1:3000/v1",
    api_key="sk-...",
)

# Chat + system + reasoning
r = client.chat(
    [ChatMessage.system("Be brief."), ChatMessage.user("hi")],
    reasoning_effort="low",
    top_p=0.9,
    seed=1,
    max_tokens=64,
)
print(r.content, r.response_id)

# Function call multi-turn
tools = [FunctionTool(
    name="lookup",
    parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
)]

def exec_tool(name, args):
    return f"value-for-{args.get('key')}"

r = client.run_tool_loop(
    [ChatMessage.user("lookup foo via tool")],
    tools=tools,
    executor=exec_tool,
    tool_choice=ToolChoice.function("lookup"),
    max_tool_calls=2,
    parallel_tool_calls=False,
)
print(r.content, r.tool_calls)

# Web search / code interpreter
print(client.chat_with_web_search("latest AI news one line").content)
print(client.chat_with_code_interpreter("compute 21*2").content)

# JSON mode
print(client.chat_json([ChatMessage.user('{"ok":true} as JSON')], schema={"type":"object","properties":{"ok":{"type":"boolean"}}}).content)

# Vision
print(client.chat_vision("describe", "data:image/png;base64,...").content)
```

## Offline matrix check

```bash
python client/python/test_matrix_offline.py
```

## Live e2e

```bash
export OPENAI_BASE_URL=http://127.0.0.1:3000/v1
export OPENAI_API_KEY=...
python client/python/e2e_matrix_live.py
```

## Honest wire contract (client → proxy → cli-chat-proxy `/responses`)

OpenAI Chat Completions is the **client-facing** surface. Upstream Grok is always
hit via **Responses** (`POST …/responses`) after translation in
`internal/upstream/grok/responses_bridge.go`.

| Matrix field | Client sends (Chat) | Proxy sanitize | Upstream `/responses` |
|---|---|---|---|
| tools function | nested `{type:function,function:{…}}` | normalize schema | **flattened** `{type:function,name,parameters}` |
| tool_choice specific | nested `{type:function,function:{name}}` | keep specific | **flattened** `{type:function,name}` (critical) |
| tool_choice auto/none/required | string | passthrough | string |
| tool result multi-turn | `role=tool` + `tool_call_id` | stabilize | `function_call_output` |
| system prompt | `role=system` / `instructions` | keep | system→`developer` input; `instructions` kept |
| vision | `image_url` parts | keep | `input_image` |
| JSON mode | `response_format` | keep | `text.format` |
| web_search / code_interpreter | builtin tools | **preserve** | passthrough (+ default `x_search` inject) |
| reasoning | `reasoning_effort` | keep | `reasoning.effort` (clamped low/medium/high) |
| top_p / temperature | yes | clamp | passthrough |
| max_tool_calls / parallel_tool_calls | yes | keep | passthrough |
| stop / seed | yes | normalize | **accepted then dropped** (Responses/cli-proxy has no seed/stop) |
| presence/frequency_penalty | yes | **stripped** | not sent (upstream rejects) |
| response_id | response field | sticky maps | bridged from Responses `id` |

### Why tool_choice flattening matters

Historical v1.9.62: nested Chat `tool_choice` against cli-chat-proxy produced
**empty 200 bodies**. Responses API expects:

```json
{"type":"function","name":"lookup"}
```

not:

```json
{"type":"function","function":{"name":"lookup"}}
```

`convertToolChoiceToResponses` now enforces the flat shape before upstream.

### Rebuild note

After pulling these proxy changes, **rebuild/restart the Go binary**. Clients alone
cannot fix a running old sanitize/bridge.
