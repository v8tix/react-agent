// Package mcpadapter bridges github.com/v8tix/mcp-toolkit with react-agent.
//
// It provides [RegistryExecutor] — a concurrent, retry-aware
// [model.ToolExecutor] backed by a mcp-toolkit [llmregistry.Registry]. Tools
// that implement observable.Tool are executed via ExecuteRx (retry +
// exponential backoff); plain handler.ExecutableTool implementations fall back
// to a simple rxgo.Defer wrapper.
//
// # Quick start
//
//	reg := registry.New(tools...)
//	defs, executor := mcpadapter.FromRegistry(reg)
//
//	a := agent.New(client, defs, executor).
//	         WithInstructions("You are a helpful assistant.").
//	         WithMaxSteps(10)
//
//	result, _, err := a.Run(ctx, "What is the weather in Paris?")
//
// # MCP tools
//
// To use tools discovered from a live MCP session:
//
//	mcpTools := mcp.NewTools(discoveredNames, session).Build()
//	reg      := registry.New(mcpTools...)
//	defs, executor := mcpadapter.FromRegistry(reg)
//
// # Concurrent execution
//
// [RegistryExecutor.Execute] fans out all tool calls concurrently via
// rxgo.Merge and restores the original call order before returning. Errors are
// encoded into ToolResult.Status="error" so the agent loop always continues
// and lets the LLM reason about any failures.
package mcpadapter
