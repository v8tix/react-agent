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
	"time"

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
	observer     Observer
}

// Option is a functional option for configuring an Agent.
type Option func(*Agent)

// WithMaxSteps overrides the default step limit (10).
// Panics if n < 1 — zero or negative steps is a programming error.
func WithMaxSteps(n int) Option {
	if n < 1 {
		panic(fmt.Sprintf("agent: WithMaxSteps: n must be >= 1, got %d", n))
	}
	return func(a *Agent) { a.maxSteps = n }
}

// WithInstructions sets the system prompt sent on every LLM request.
func WithInstructions(s string) Option {
	return func(a *Agent) { a.instructions = s }
}

// WithObserver registers an Observer that receives lifecycle hooks for every run.
// Passing nil is a no-op (the default NoopObserver remains in place).
func WithObserver(obs Observer) Option {
	return func(a *Agent) {
		if obs != nil {
			a.observer = obs
		}
	}
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
		observer:  NoopObserver{},
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

	a.observer.OnRunStart(ctx, execCtx.id, userMessage)

	for execCtx.currentStep < a.maxSteps {
		if err := a.Step(ctx, execCtx); err != nil {
			err = fmt.Errorf("agent step %d: %w", execCtx.currentStep, err)
			a.observer.OnRunEnd(ctx, execCtx.id, nil, err)
			return nil, err
		}
		if execCtx.Done() {
			break
		}
		execCtx.IncrementStep()
	}

	if !execCtx.Done() {
		err := fmt.Errorf("%w after %d steps", ErrMaxStepsReached, a.maxSteps)
		a.observer.OnRunEnd(ctx, execCtx.id, nil, err)
		return nil, err
	}

	output, ok := execCtx.FinalResult()
	if !ok {
		err := fmt.Errorf("agent: internal error — finalResult type is %T, expected string", execCtx.finalResult)
		a.observer.OnRunEnd(ctx, execCtx.id, nil, err)
		return nil, err
	}

	result := &Result{
		Output:     output,
		ToolCalled: anyToolCalled(execCtx),
		Context:    execCtx,
	}
	a.observer.OnRunEnd(ctx, execCtx.id, result, nil)
	return result, nil
}

// Step executes one Think → (optionally) Act cycle, mutating execCtx in place.
// It is exported so callers can drive the loop manually for streaming,
// checkpointing, or human-in-the-loop interrupts. Use execCtx.Done() to check
// whether the agent produced a final answer.
func (a *Agent) Step(ctx context.Context, execCtx *ExecutionContext) error {
	a.observer.OnStepStart(ctx, execCtx.id, execCtx.currentStep)

	resp, err := a.Think(ctx, execCtx)
	if err != nil {
		a.observer.OnStepEnd(ctx, execCtx.id, execCtx.currentStep, err)
		return err
	}

	toolCalls := collectToolCalls(resp.Content)
	if len(toolCalls) == 0 {
		if msg := extractAssistantMessage(resp.Content); msg != "" {
			execCtx.AddEvent("agent", model.Message{Role: "assistant", Content: msg})
			execCtx.setFinalResult(msg)
		}
		a.observer.OnStepEnd(ctx, execCtx.id, execCtx.currentStep, nil)
		return nil
	}

	err = a.Act(ctx, execCtx, toolCalls)
	a.observer.OnStepEnd(ctx, execCtx.id, execCtx.currentStep, err)
	return err
}

// Think calls the LLM with the current execution context and returns its response.
func (a *Agent) Think(ctx context.Context, execCtx *ExecutionContext) (model.Response, error) {
	req := a.prepareRequest(execCtx)
	start := time.Now()
	resp, err := a.llmClient.Generate(ctx, req)
	a.observer.OnLLMCall(ctx, execCtx.id, req, resp, time.Since(start), err)
	return resp, err
}

// Act executes all requested tool calls via ToolExecutor and records the results.
// The agent's tool-call decision is appended as an "agent" event BEFORE execution,
// then tool results are appended as a "tools" event AFTER execution.
func (a *Agent) Act(ctx context.Context, execCtx *ExecutionContext, calls []model.ToolCall) error {
	callItems := make([]model.ContentItem, len(calls))
	for i, tc := range calls {
		callItems[i] = tc
	}
	// Record the tool-call decision before attempting execution so the event log
	// is accurate even if the executor is nil or returns an error.
	execCtx.AddEvent("agent", callItems...)

	if a.executor == nil {
		err := fmt.Errorf("agent: cannot execute tools — ToolExecutor is nil")
		a.observer.OnToolExecution(ctx, execCtx.id, calls, nil, 0, err)
		return err
	}

	start := time.Now()
	results, err := a.executor.Execute(ctx, calls)
	a.observer.OnToolExecution(ctx, execCtx.id, calls, results, time.Since(start), err)
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
