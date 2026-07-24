package server_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/hm2899/grokcli-2api/internal/auth"
	"github.com/hm2899/grokcli-2api/internal/config"
	"github.com/hm2899/grokcli-2api/internal/pool"
	"github.com/hm2899/grokcli-2api/internal/server"
)

// fakeUpstreamToolCall returns an SSE stream that emits a single tool_call
// for the given tool name and arguments JSON. The upstream (Grok /responses
// via cli-chat-proxy) always emits internal-form args (file_path). The proxy
// must project them to the client's schema key (path vs file_path).
func fakeUpstreamToolCall(t *testing.T, toolName, argsJSON string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flush := func() {
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		base := map[string]any{"id": "chatcmpl_e2e", "object": "chat.completion.chunk", "created": 1700000000, "model": "grok-4.5"}
		write := func(m map[string]any) {
			b, _ := json.Marshal(merge(base, m))
			_, _ = io.WriteString(w, "data: "+string(b)+"\n\n")
			flush()
		}
		write(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "id": "call_e2e", "type": "function",
			"function": map[string]any{"name": toolName, "arguments": argsJSON},
		}}}, "finish_reason": nil}}})
		write(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}}, "usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		flush()
	})
}

// TestPathSchemaAgentToolCallRoundTrip verifies that an agent declaring tools
// with "path" (Hermes write_file, generic OpenAI) receives outbound
// tool_calls with "path" — not the internal "file_path" that the upstream
// (Grok) emitted. This is the critical agent-tool-calling correctness test.
func TestPathSchemaAgentToolCallRoundTrip(t *testing.T) {
	// Upstream emits file_path (internal form after Grok normalize).
	upstream := httptest.NewServer(fakeUpstreamToolCall(t, "write_file", `{"file_path":"/tmp/demo.txt","content":"hello"}`))
	defer upstream.Close()

	h := server.NewMux(server.Options{
		Ready:       func() bool { return true },
		ChatEnabled: true,
		APIKeys:     auth.NewAPIKeyVerifier(config.Config{LegacyAPIKey: "secret"}, nil),
		Candidates:  []pool.Candidate{{ID: "acc", Token: "tok", Enabled: true}},
		Config: config.Config{
			UpstreamBase: upstream.URL + "/v1",
			DefaultModel: "grok-4.5",
			SSEKeepalive: 4 * time.Second,
		},
	})

	tools := `[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]`

	t.Run("stream", func(t *testing.T) {
		body := `{"model":"grok-4.5","stream":true,"tools":` + tools + `,"messages":[{"role":"user","content":"write a file"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer secret")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		out := rec.Body.String()
		if rec.Code != 200 {
			t.Fatalf("status=%d body=%s", rec.Code, out)
		}
		// Must have "path" key with the file path value.
		if !strings.Contains(out, `"path"`) && !strings.Contains(out, `\"path\"`) {
			t.Fatalf("path-schema agent did not receive path key:\n%s", out)
		}
		// Must NOT have file_path leaked to the client.
		if strings.Contains(out, `"file_path"`) || strings.Contains(out, `\"file_path\"`) {
			t.Fatalf("file_path leaked to path-schema agent:\n%s", out)
		}
		// Must contain the actual path value.
		if !strings.Contains(out, "/tmp/demo.txt") {
			t.Fatalf("path value missing:\n%s", out)
		}
	})

	t.Run("non-stream", func(t *testing.T) {
		body := `{"model":"grok-4.5","tools":` + tools + `,"messages":[{"role":"user","content":"write a file"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer secret")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
			t.Fatal(err)
		}
		choices, _ := payload["choices"].([]any)
		if len(choices) == 0 {
			t.Fatalf("no choices: %#v", payload)
		}
		msg := choices[0].(map[string]any)["message"].(map[string]any)
		toolCalls, _ := msg["tool_calls"].([]any)
		if len(toolCalls) == 0 {
			t.Fatalf("no tool_calls: %#v", msg)
		}
		tc := toolCalls[0].(map[string]any)
		fn := tc["function"].(map[string]any)
		args := fn["arguments"].(string)
		var argsObj map[string]any
		if err := json.Unmarshal([]byte(args), &argsObj); err != nil {
			t.Fatalf("args not JSON: %s err=%v", args, err)
		}
		if argsObj["path"] != "/tmp/demo.txt" {
			t.Fatalf("path missing or wrong: %#v", argsObj)
		}
		if _, ok := argsObj["file_path"]; ok {
			t.Fatalf("file_path leaked: %s", args)
		}
		if argsObj["content"] != "hello" {
			t.Fatalf("content lost: %s", args)
		}
	})
}

// TestClaudeCodeFilePathSchemaUnchanged verifies that Claude Code (file_path
// schema) still receives file_path — the path projection must not break
// existing Claude Code agents.
func TestClaudeCodeFilePathSchemaUnchanged(t *testing.T) {
	upstream := httptest.NewServer(fakeUpstreamToolCall(t, "Write", `{"file_path":"/tmp/a.go","content":"x"}`))
	defer upstream.Close()

	h := server.NewMux(server.Options{
		Ready:       func() bool { return true },
		ChatEnabled: true,
		APIKeys:     auth.NewAPIKeyVerifier(config.Config{LegacyAPIKey: "secret"}, nil),
		Candidates:  []pool.Candidate{{ID: "acc", Token: "tok", Enabled: true}},
		Config: config.Config{
			UpstreamBase: upstream.URL + "/v1",
			DefaultModel: "grok-4.5",
			SSEKeepalive: 4 * time.Second,
		},
	})

	// Claude Code Write tool schema: file_path
	tools := `[{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}}]`
	body := `{"model":"grok-4.5","stream":true,"tools":` + tools + `,"messages":[{"role":"user","content":"write"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	out := rec.Body.String()
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, out)
	}
	if !strings.Contains(out, `"file_path"`) && !strings.Contains(out, `\"file_path\"`) {
		t.Fatalf("Claude Code file_path schema did not receive file_path:\n%s", out)
	}
}

// TestVisionImageURLRoundTrip verifies image_url content blocks survive the
// full NewMux → upstream path and are present in the request the upstream
// receives.
func TestVisionImageURLRoundTrip(t *testing.T) {
	var upstreamSawImage bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(string(raw), "image_url") || strings.Contains(string(raw), "input_image") || strings.Contains(string(raw), "data:image") {
			upstreamSawImage = true
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := server.NewMux(server.Options{
		Ready:       func() bool { return true },
		ChatEnabled: true,
		APIKeys:     auth.NewAPIKeyVerifier(config.Config{LegacyAPIKey: "secret"}, nil),
		Candidates:  []pool.Candidate{{ID: "acc", Token: "tok", Enabled: true}},
		Config: config.Config{
			UpstreamBase: upstream.URL + "/v1",
			DefaultModel: "grok-4.5",
			SSEKeepalive: 4 * time.Second,
		},
	})

	body := `{"model":"grok-4.5","messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KG="}}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !upstreamSawImage {
		t.Fatalf("upstream did not receive image: %s", rec.Body.String())
	}
}

// TestResponseFormatPassThrough verifies json_object and json_schema
// response_format reach the upstream request body.
func TestResponseFormatPassThrough(t *testing.T) {
	var sawJSONObject, sawJSONSchema bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		s := string(raw)
		if strings.Contains(s, "json_object") {
			sawJSONObject = true
		}
		if strings.Contains(s, "json_schema") {
			sawJSONSchema = true
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := server.NewMux(server.Options{
		Ready:       func() bool { return true },
		ChatEnabled: true,
		APIKeys:     auth.NewAPIKeyVerifier(config.Config{LegacyAPIKey: "secret"}, nil),
		Candidates:  []pool.Candidate{{ID: "acc", Token: "tok", Enabled: true}},
		Config: config.Config{
			UpstreamBase: upstream.URL + "/v1",
			DefaultModel: "grok-4.5",
			SSEKeepalive: 4 * time.Second,
		},
	})

	t.Run("json_object", func(t *testing.T) {
		body := `{"model":"grok-4.5","response_format":{"type":"json_object"},"messages":[{"role":"user","content":"return json"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer secret")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		if !sawJSONObject {
			t.Fatal("upstream did not receive json_object response_format")
		}
	})

	t.Run("json_schema", func(t *testing.T) {
		body := `{"model":"grok-4.5","response_format":{"type":"json_schema","json_schema":{"name":"my_schema","schema":{"type":"object","properties":{"x":{"type":"string"}}}}},"messages":[{"role":"user","content":"return schema json"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer secret")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		if !sawJSONSchema {
			t.Fatal("upstream did not receive json_schema response_format")
		}
	})
}

// TestToolResultMultiTurn verifies a tool result message (role=tool) is
// accepted and forwarded to upstream as function_call_output (Responses API)
// or as a normal OpenAI tool message — proving multi-turn tool conversations work.
func TestToolResultMultiTurn(t *testing.T) {
	var upstreamSawToolResult bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		s := string(raw)
		// Chat path forwards role=tool as-is or as function_call_output on Responses.
		if strings.Contains(s, "function_call_output") || strings.Contains(s, `"role":"tool"`) {
			upstreamSawToolResult = true
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"result: 42"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := server.NewMux(server.Options{
		Ready:       func() bool { return true },
		ChatEnabled: true,
		APIKeys:     auth.NewAPIKeyVerifier(config.Config{LegacyAPIKey: "secret"}, nil),
		Candidates:  []pool.Candidate{{ID: "acc", Token: "tok", Enabled: true}},
		Config: config.Config{
			UpstreamBase: upstream.URL + "/v1",
			DefaultModel: "grok-4.5",
			SSEKeepalive: 4 * time.Second,
		},
	})

	// Full multi-turn: user → assistant tool_call → tool result → assistant text
	body := `{"model":"grok-4.5","messages":[
		{"role":"user","content":"what is 6*7"},
		{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"a\":6,\"b\":7}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"42"}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !upstreamSawToolResult {
		t.Fatalf("upstream did not receive tool result: status=%d body=%s", rec.Code, rec.Body.String())
	}
}
