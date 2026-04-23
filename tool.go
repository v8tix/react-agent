package agent

// ToolDefinition describes a function the LLM can invoke.
// Populate this from your tool registry or define it directly.
// This type is intentionally independent of any specific tool framework —
// convert from mcp-toolkit, langchain, or your own registry with a simple loop.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	// Parameters is a JSON Schema object describing the function arguments.
	// Example: map[string]any{"type": "object", "properties": {...}, "required": [...]}
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}
