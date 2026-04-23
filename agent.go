// Package agent implements the ReAct (Reason + Act) pattern for AI agents.
//
// A ReAct agent runs a bounded Think → Act → Observe loop: the model thinks
// (generates a response), acts (calls tools), and observes (receives results),
// repeating until it produces a final answer or exhausts its step limit.
//
// # Quick start
//
//	client   := agent.NewLiteLLMClient(openaiClient, model)
//	executor := myToolExecutor  // implements model.ToolExecutor
//	a        := agent.New(client, toolDefs, executor,
//	                agent.WithInstructions("You are a helpful assistant."),
//	                agent.WithMaxSteps(10))
//
//	result, err := a.Run(ctx, "Who won the 2025 Nobel Prize in Physics?")
//	fmt.Println(result.Output)
//
// # Execution history
//
// Every run produces a full event log in result.Context.Events(). Each event
// has an Author ("user", "agent", or "tools"), a timestamp, and typed content
// items (model.Message, model.ToolCall, model.ToolResult).
//
// # Bringing your own tools
//
// Implement model.ToolExecutor to plug in any dispatch strategy. A typical
// adapter converts []model.ToolCall to your tool-runner format, executes
// concurrently, and returns []model.ToolResult.
package agent

import (
	"context"
	"fmt"

	"github.com/v8tix/react-agent/model"
)

// Agent is the ReAct orchestrator. It runs a Think → Act → Observe loop until
// the LLM produces a final answer or maxSteps is exhausted.
type Agent struct {
	llmClient    LLMClient
	toolDefs     []model.ToolDefinition // definitions sent to the LLM each turn
	executor     model.ToolExecutor     // nil is safe when toolDefs is empty
	instructions string
	maxSteps     int
}

// Option is a functional option for configuring an Agent.
type Option func(*Agent)

// WithMaxSteps overrides the default step limit (10).
func WithMaxSteps(n int) Option {
	return func(a *Agent) { a.maxSteps = n }
}

// WithInstructions sets the system prompt sent on every LLM request.
func WithInstructions(s string) Option {
	return func(a *Agent) { a.instructions = s }
}

// New creates an Agent.
//   - defs: tool definitions the LLM can call (pass nil or empty for no tools)
//   - executor: executes tool calls concurrently (pass nil when defs is empty)
func New(client LLMClient, defs []model.ToolDefinition, executor model.ToolExecutor, opts ...Option) *Agent {
	a := &Agent{
		llmClient: client,
		toolDefs:  defs,
		executor:  executor,
		maxSteps:  10,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Run executes the full ReAct loop for a single user message.
// It returns a Result with the final answer, whether any tool was called,
// and the complete ExecutionContext containing the event history.
func (a *Agent) Run(ctx context.Context, userMessage string) (*Result, error) {
	execCtx := newExecutionContext()
	execCtx.AddEvent("user", model.Message{Role: "user", Content: userMessage})

	for execCtx.CurrentStep < a.maxSteps {
		if err := a.Step(ctx, execCtx); err != nil {
			return nil, fmt.Errorf("agent step %d: %w", execCtx.CurrentStep, err)
		}
		if execCtx.finalResult != nil {
			break
		}
		execCtx.IncrementStep()
	}

	if execCtx.finalResult == nil {
		return nil, fmt.Errorf("%w after %d steps", ErrMaxStepsReached, a.maxSteps)
	}

	return &Result{
		Output:     execCtx.finalResult.(string),
		ToolCalled: anyToolCalled(execCtx),
		Context:    execCtx,
	}, nil
}

// Step executes one Think → (optionally) Act cycle, mutating execCtx in place.
// It is exported so callers can drive the loop manually for streaming,
// checkpointing, or human-in-the-loop interrupts.
func (a *Agent) Step(ctx context.Context, execCtx *ExecutionContext) error {
	resp, err := a.Think(ctx, execCtx)
	if err != nil {
		return err
	}

	toolCalls := collectToolCalls(resp.Content)
	if len(toolCalls) == 0 {
		if msg := extractAssistantMessage(resp.Content); msg != "" {
			execCtx.finalResult = msg
		}
		return nil
	}

	return a.Act(ctx, execCtx, toolCalls)
}

// Think calls the LLM with the current execution context and returns its response.
func (a *Agent) Think(ctx context.Context, execCtx *ExecutionContext) (model.Response, error) {
	return a.llmClient.Generate(ctx, a.prepareRequest(execCtx))
}

// Act executes all requested tool calls via ToolExecutor and records the results.
// The agent's tool-call decision is appended as an "agent" event BEFORE execution,
// then tool results are appended as a "tools" event AFTER execution.
func (a *Agent) Act(ctx context.Context, execCtx *ExecutionContext, calls []model.ToolCall) error {
	callItems := make([]model.ContentItem, len(calls))
	for i, tc := range calls {
		callItems[i] = tc
	}
	execCtx.AddEvent("agent", callItems...)

	results, err := a.executor.Execute(ctx, calls)
	if err != nil {
		return fmt.Errorf("act execute: %w", err)
	}

	resultItems := make([]model.ContentItem, len(results))
	for i, tr := range results {
		resultItems[i] = tr
	}
	execCtx.AddEvent("tools", resultItems...)
	return nil
}

func (a *Agent) prepareRequest(execCtx *ExecutionContext) model.Request {
	return model.Request{
		Instructions: a.instructions,
		Events:       execCtx.Events(),
		Tools:        a.toolDefs,
	}
}

func extractAssistantMessage(items []model.ContentItem) string {
	for _, item := range items {
		if m, ok := item.(model.Message); ok && m.Role == "assistant" {
			return m.Content
		}
	}
	return ""
}

func anyToolCalled(execCtx *ExecutionContext) bool {
	for _, event := range execCtx.Events() {
		if event.Author == "agent" {
			for _, item := range event.Content {
				if _, ok := item.(model.ToolCall); ok {
					return true
				}
			}
		}
	}
	return false
}
