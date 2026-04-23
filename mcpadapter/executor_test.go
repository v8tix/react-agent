package mcpadapter_test

import (
	"context"
	"encoding/json"
	"testing"

	llmhandler "github.com/v8tix/mcp-toolkit/handler"
	llmmodel "github.com/v8tix/mcp-toolkit/model"
	llmregistry "github.com/v8tix/mcp-toolkit/registry"

	agent "github.com/v8tix/react-agent"
	"github.com/v8tix/react-agent/mcpadapter"
)

// ─── test tool ────────────────────────────────────────────────────────────────

type echoArgs struct {
	Message string `json:"message"`
}

func newEchoTool() llmhandler.ExecutableTool {
	return llmhandler.NewTool("echo", "Echoes the input message back.",
		func(_ context.Context, in echoArgs) (string, error) {
			return "echo: " + in.Message, nil
		},
	)
}

func newFailTool() llmhandler.ExecutableTool {
	return llmhandler.NewTool("fail", "Always returns an error.",
		func(_ context.Context, _ echoArgs) (string, error) {
			return "", context.DeadlineExceeded
		},
	)
}

// ─── Defs ─────────────────────────────────────────────────────────────────────

func TestDefs_ConvertsToolDefinitions(t *testing.T) {
	reg := llmregistry.New(newEchoTool())
	defs := mcpadapter.Defs(reg.All())

	if len(defs) != 1 {
		t.Fatalf("want 1 def, got %d", len(defs))
	}
	if defs[0].Name != "echo" {
		t.Errorf("want name=echo, got %s", defs[0].Name)
	}
	if defs[0].Description == "" {
		t.Error("expected non-empty description")
	}
	if defs[0].Parameters == nil {
		t.Error("expected non-nil parameters")
	}
}

func TestDefs_EmptyRegistry(t *testing.T) {
	defs := mcpadapter.Defs([]llmmodel.ToolDefinition{})
	if len(defs) != 0 {
		t.Fatalf("want empty, got %d", len(defs))
	}
}

// ─── FromRegistry ─────────────────────────────────────────────────────────────

func TestFromRegistry_ReturnsBothDefsAndExecutor(t *testing.T) {
	reg := llmregistry.New(newEchoTool())
	defs, executor := mcpadapter.FromRegistry(reg)

	if len(defs) != 1 {
		t.Fatalf("want 1 def, got %d", len(defs))
	}
	if executor == nil {
		t.Fatal("executor should not be nil")
	}
}

// ─── RegistryExecutor.Execute ─────────────────────────────────────────────────

func TestRegistryExecutor_Execute_SingleCall_Success(t *testing.T) {
	reg := llmregistry.New(newEchoTool())
	executor := mcpadapter.NewRegistryExecutor(reg)

	args, _ := json.Marshal(echoArgs{Message: "hello"})
	results, err := executor.Execute(context.Background(), []agent.ToolCall{
		{ID: "call-1", Name: "echo", Arguments: args},
	})

	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if results[0].Status != "success" {
		t.Errorf("want success, got %s", results[0].Status)
	}
	if len(results[0].Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	// Content is JSON-encoded, so "echo: hello" → `"echo: hello"`
	var got string
	if err := json.Unmarshal([]byte(results[0].Content[0]), &got); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if got != "echo: hello" {
		t.Errorf("want 'echo: hello', got %q", got)
	}
}

func TestRegistryExecutor_Execute_ToolNotFound_ReturnsError(t *testing.T) {
	reg := llmregistry.New(newEchoTool())
	executor := mcpadapter.NewRegistryExecutor(reg)

	results, err := executor.Execute(context.Background(), []agent.ToolCall{
		{ID: "call-1", Name: "nonexistent", Arguments: json.RawMessage(`{}`)},
	})

	if err != nil {
		t.Fatal("Execute should not return a Go error for unknown tools")
	}
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if results[0].Status != "error" {
		t.Errorf("want error status for unknown tool, got %s", results[0].Status)
	}
}

func TestRegistryExecutor_Execute_EmptyCalls(t *testing.T) {
	reg := llmregistry.New(newEchoTool())
	executor := mcpadapter.NewRegistryExecutor(reg)

	results, err := executor.Execute(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Fatalf("want 0 results, got %d", len(results))
	}
}

func TestRegistryExecutor_Execute_MultipleCalls_OrderPreserved(t *testing.T) {
	// Two echo tools registered; we call them in a specific order and verify
	// the results come back in the same order despite concurrent execution.
	reg := llmregistry.New(newEchoTool())
	executor := mcpadapter.NewRegistryExecutor(reg)

	calls := []agent.ToolCall{
		{ID: "c1", Name: "echo", Arguments: mustMarshal(echoArgs{Message: "first"})},
		{ID: "c2", Name: "echo", Arguments: mustMarshal(echoArgs{Message: "second"})},
		{ID: "c3", Name: "echo", Arguments: mustMarshal(echoArgs{Message: "third"})},
	}

	results, err := executor.Execute(context.Background(), calls)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("want 3 results, got %d", len(results))
	}

	expected := []string{"echo: first", "echo: second", "echo: third"}
	for i, r := range results {
		var got string
		if err := json.Unmarshal([]byte(r.Content[0]), &got); err != nil {
			t.Fatalf("result %d: unmarshal: %v", i, err)
		}
		if got != expected[i] {
			t.Errorf("result %d: want %q, got %q", i, expected[i], got)
		}
	}
}

func TestRegistryExecutor_Execute_ToolError_EncodedInResult(t *testing.T) {
	reg := llmregistry.New(newFailTool())
	executor := mcpadapter.NewRegistryExecutor(reg)

	results, err := executor.Execute(context.Background(), []agent.ToolCall{
		{ID: "c1", Name: "fail", Arguments: mustMarshal(echoArgs{Message: "x"})},
	})

	if err != nil {
		t.Fatal("tool execution error should be encoded in result, not returned as Go error")
	}
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if results[0].Status != "error" {
		t.Errorf("want error status, got %s", results[0].Status)
	}
}

func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
