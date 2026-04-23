package model

import "context"

// ToolDefinition describes a function the LLM can invoke.
// This type is intentionally independent of any specific tool framework —
// convert from mcp-toolkit, langchain, or your own registry with a simple loop.
type ToolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Parameters is a JSON Schema object describing the function arguments.
	// Example: map[string]any{"type": "object", "properties": {...}, "required": [...]}
	Parameters map[string]any `json:"parameters"`
	Strict     bool           `json:"strict,omitempty"`
}

// ToolExecutor abstracts concurrent tool dispatch for the Agent.
// Implement this interface to connect any tool-running backend.
// The default adapter for mcp-toolkit is in the mcpadapter subpackage.
type ToolExecutor interface {
	Execute(ctx context.Context, calls []ToolCall) ([]ToolResult, error)
}
