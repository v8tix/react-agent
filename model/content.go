package model

import "encoding/json"

// ContentItem is a closed discriminated union of all values that can appear
// in an Event. The unexported marker method prevents external packages from
// satisfying the interface, keeping type switches exhaustive.
type ContentItem interface {
	contentItem()
	Type() string
}

// Message is a chat turn (system / user / assistant plain text).
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (Message) contentItem() {}
func (Message) Type() string  { return "message" }

// ToolCall is a single tool invocation requested by the LLM.
type ToolCall struct {
	ID        string          `json:"tool_call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func (ToolCall) contentItem() {}
func (ToolCall) Type() string  { return "tool_call" }

// ToolResult is the outcome of executing a ToolCall.
type ToolResult struct {
	ID      string   `json:"tool_call_id"`
	Name    string   `json:"name"`
	Status  string   `json:"status"` // "success" | "error"
	Content []string `json:"content"`
}

func (ToolResult) contentItem() {}
func (ToolResult) Type() string  { return "tool_result" }
