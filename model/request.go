package model

// Request is the agent's view of a single LLM call.
// Events carries the full conversation history; the LLMClient translates
// each Event into the provider-specific message format.
type Request struct {
	Instructions string
	Events       []Event
	Tools        []ToolDefinition
	MaxTokens    int64
}

// Response is the agent's view of a single LLM reply.
// Content contains either ToolCall items (when the model wants to act) or a
// single Message item with Role "assistant" (when the model has an answer).
type Response struct {
	Content []ContentItem
}
