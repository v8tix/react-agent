// Package model contains the core data types shared across react-agent and its
// sub-packages.
//
// The types here are intentionally minimal and framework-agnostic so that the
// parent package imports only what it needs, and external adapters can convert
// to/from their own representations without depending on the orchestration
// logic.
//
// # Key types
//
//   - [ToolDefinition] — JSON-schema description of a function the LLM can call.
//   - [ToolExecutor] — interface for dispatching [ToolCall] slices to a backend.
//   - [ContentItem] — sealed discriminated union: [Message], [ToolCall], [ToolResult].
//   - [Event] — a timestamped, authored entry in an [ExecutionContext] history.
//   - [Request] / [Response] — the exchange contract between Agent and [LLMClient].
package model
