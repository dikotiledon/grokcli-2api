package toolcall

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPreferredPathArgKeyFromSchemas(t *testing.T) {
	// Hermes / OpenAI-style write_file: path + content
	if got := PreferredPathArgKey(map[string]any{
		"type":       "object",
		"properties": map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}},
		"required":   []any{"path", "content"},
	}); got != "path" {
		t.Fatalf("hermes write_file schema → %q", got)
	}
	// Claude Code Write: file_path + content
	if got := PreferredPathArgKey(map[string]any{
		"type":       "object",
		"properties": map[string]any{"file_path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}},
		"required":   []any{"file_path", "content"},
	}); got != "file_path" {
		t.Fatalf("claude write schema → %q", got)
	}
	// Missing schema defaults to Claude-compatible file_path
	if got := PreferredPathArgKey(nil); got != "file_path" {
		t.Fatalf("nil schema → %q", got)
	}
}

func TestPathArgKeyMapFromTools(t *testing.T) {
	tools := []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "write_file",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}},
					"required":   []any{"path"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "Read",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"file_path": map[string]any{"type": "string"}},
					"required":   []any{"file_path"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "get_time",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"tz": map[string]any{"type": "string"}},
				},
			},
		},
	}
	m := PathArgKeyMap(tools)
	if m["write_file"] != "path" {
		t.Fatalf("write_file=%q map=%#v", m["write_file"], m)
	}
	if m["Read"] != "file_path" {
		t.Fatalf("Read=%q map=%#v", m["Read"], m)
	}
	if _, ok := m["get_time"]; ok {
		t.Fatalf("non-path tool should not be mapped: %#v", m)
	}
}

func TestProjectPathArgsForClientHermesWriteFile(t *testing.T) {
	// Internal form after normalize: file_path + content
	internal := `{"file_path":"D:/tmp/demo.txt","content":"hello"}`
	out := ProjectPathArgsForClient(internal, "write_file", "path")
	var obj map[string]any
	if err := json.Unmarshal([]byte(out), &obj); err != nil {
		t.Fatalf("json: %v out=%s", err, out)
	}
	if obj["path"] != "D:/tmp/demo.txt" {
		t.Fatalf("path missing: %s", out)
	}
	if _, ok := obj["file_path"]; ok {
		t.Fatalf("file_path leaked to path-schema client: %s", out)
	}
	if obj["content"] != "hello" {
		t.Fatalf("content lost: %s", out)
	}
}

func TestProjectPathArgsForClientClaudeUnchanged(t *testing.T) {
	internal := `{"file_path":"/tmp/a.go","content":"x"}`
	out := ProjectPathArgsForClient(internal, "Write", "file_path")
	if !strings.Contains(out, `"file_path"`) {
		t.Fatalf("claude path should stay file_path: %s", out)
	}
	if strings.Contains(out, `"path"`) && !strings.Contains(out, `"file_path"`) {
		t.Fatalf("unexpected projection: %s", out)
	}
}

func TestDefaultPathArgKeyKnownAgents(t *testing.T) {
	if DefaultPathArgKey("write_file") != "path" {
		t.Fatalf("write_file default")
	}
	if DefaultPathArgKey("read_file") != "path" {
		t.Fatalf("read_file default")
	}
	if DefaultPathArgKey("Write") != "file_path" {
		t.Fatalf("Write default claude")
	}
	if DefaultPathArgKey("Read") != "file_path" {
		t.Fatalf("Read default claude")
	}
}

func TestProjectPathArgsFromGrokPathAlias(t *testing.T) {
	// Grok often emits path= even for Claude tools; internal normalize → file_path.
	// Hermes client still needs path= back.
	raw := `{"path":"C:/Users/x/a.txt","content":"z"}`
	// Simulate normalize then project for path-schema client.
	norm := NormalizeJSON(raw, "write_file")
	out := ProjectPathArgsForClient(norm, "write_file", "path")
	if !strings.Contains(out, `"path"`) {
		t.Fatalf("expected path key: norm=%s out=%s", norm, out)
	}
	if strings.Contains(out, `"file_path"`) {
		t.Fatalf("file_path leaked: %s", out)
	}
}


// TestPreferredPathArgKeyCamelCase verifies camelCase variants are detected.
func TestPreferredPathArgKeyCamelCase(t *testing.T) {
	// OpenCode/Cursor: filePath (camelCase)
	if got := PreferredPathArgKey(map[string]any{
		"type":       "object",
		"properties": map[string]any{"filePath": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}},
		"required":   []any{"filePath", "content"},
	}); got != "filePath" {
		t.Fatalf("camelCase filePath → %q", got)
	}
	// Mixed: both path and filePath present, required has filePath first
	if got := PreferredPathArgKey(map[string]any{
		"type":       "object",
		"properties": map[string]any{"path": map[string]any{"type": "string"}, "filePath": map[string]any{"type": "string"}},
		"required":   []any{"filePath"},
	}); got != "filePath" {
		t.Fatalf("required filePath should win → %q", got)
	}
}

// TestProjectPathArgsForClientFilePath projects to camelCase filePath.
func TestProjectPathArgsForClientFilePath(t *testing.T) {
	internal := `{"file_path":"C:/tmp/test.go","content":"x"}`
	out := ProjectPathArgsForClient(internal, "write", "filePath")
	if !strings.Contains(out, `"filePath"`) {
		t.Fatalf("expected filePath key: %s", out)
	}
	if strings.Contains(out, `"file_path"`) {
		t.Fatalf("file_path leaked: %s", out)
	}
	if !strings.Contains(out, "C:/tmp/test.go") {
		t.Fatalf("path value lost: %s", out)
	}
}

// TestProjectPathArgsForClientFilePathLower projects to lowercase filepath.
func TestProjectPathArgsForClientFilePathLower(t *testing.T) {
	internal := `{"file_path":"/tmp/a.go","content":"y"}`
	out := ProjectPathArgsForClient(internal, "write", "filepath")
	if !strings.Contains(out, `"filepath"`) {
		t.Fatalf("expected filepath key: %s", out)
	}
	if strings.Contains(out, `"file_path"`) {
		t.Fatalf("file_path leaked: %s", out)
	}
}

// TestProjectPathArgsForClientGrokEmitsCamelCase handles Grok emitting filePath
// directly (not just file_path).
func TestProjectPathArgsForClientGrokEmitsCamelCase(t *testing.T) {
	// Grok might emit any variant; internal normalize maps them all to file_path.
	// But if normalize hasn't run yet, ProjectPathArgsForClient should still find it.
	raw := `{"filePath":"/tmp/b.go","content":"z"}`
	out := ProjectPathArgsForClient(raw, "write", "filePath")
	if !strings.Contains(out, `"filePath"`) {
		t.Fatalf("expected filePath preserved: %s", out)
	}
	if !strings.Contains(out, "/tmp/b.go") {
		t.Fatalf("path value lost: %s", out)
	}
}

// TestProjectPathArgsForClientFilePathFromGrokPathAlias handles Grok emitting
// "path" when the client wants "filePath".
func TestProjectPathArgsForClientFilePathFromGrokPathAlias(t *testing.T) {
	raw := `{"path":"D:/work/c.go","content":"w"}`
	norm := NormalizeJSON(raw, "write_file")
	out := ProjectPathArgsForClient(norm, "write_file", "filePath")
	if !strings.Contains(out, `"filePath"`) {
		t.Fatalf("expected filePath from path alias: norm=%s out=%s", norm, out)
	}
	if strings.Contains(out, `"file_path"`) || strings.Contains(out, `"path"`) {
		t.Fatalf("old path key leaked: %s", out)
	}
	if !strings.Contains(out, "D:/work/c.go") {
		t.Fatalf("path value lost: %s", out)
	}
}
