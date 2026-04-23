package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/reactivex/rxgo/v2"
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

// New creates an Agent with sensible defaults (maxSteps=10).
//   - defs: tool definitions the LLM can call (pass nil or empty for no tools)
//   - executor: executes tool calls concurrently (pass nil when defs is empty)
//
// Use the builder methods to customise the agent:
//
//	agent.New(client, defs, executor).
//	    WithInstructions("You are helpful.").
//	    WithMaxSteps(15)
func New(client LLMClient, defs []model.ToolDefinition, executor model.ToolExecutor) *Agent {
	return &Agent{
		llmClient: client,
		toolDefs:  defs,
		executor:  executor,
		maxSteps:  10,
	}
}

// WithInstructions sets the system prompt sent on every LLM request.
func (a *Agent) WithInstructions(s string) *Agent {
	a.instructions = s
	return a
}

// WithMaxSteps overrides the default step limit (10).
// Panics if n < 1 — zero or negative steps is a programming error.
func (a *Agent) WithMaxSteps(n int) *Agent {
	if n < 1 {
		panic(fmt.Sprintf("agent: WithMaxSteps: n must be >= 1, got %d", n))
	}
	a.maxSteps = n
	return a
}

// Run executes the full ReAct loop for a single user message.
// It returns a Result, a replayable Observable of AgentEvents, and any error.
//
// The Observable is a cold, replayable stream: each call to Observe() replays
// all events from the completed run. It is safe for multiple subscribers.
func (a *Agent) Run(ctx context.Context, userMessage string) (*Result, rxgo.Observable, error) {
	// Buffer is sized to hold all possible events: 1 RunStart + maxSteps*(StepStart
	// + LLMCall + ToolExec + StepEnd) + 1 RunEnd — never drops.
	ch := make(chan AgentEvent, a.maxSteps*4+8)
	result, err := a.runCore(ctx, userMessage, ch)
	close(ch)

	// Drain the closed buffered channel into a slice to produce a replayable
	// cold observable. Each Observe() call replays all events from scratch.
	collected := make([]AgentEvent, 0, len(ch))
	for e := range ch {
		collected = append(collected, e)
	}

	obs := rxgo.Defer([]rxgo.Producer{
		func(_ context.Context, out chan<- rxgo.Item) {
			for _, e := range collected {
				out <- rxgo.Of(e)
			}
		},
	})
	return result, obs, err
}

// runCore is the internal loop. It writes AgentEvents to ch (never closes it).
func (a *Agent) runCore(ctx context.Context, userMessage string, ch chan<- AgentEvent) (*Result, error) {
	execCtx := newExecutionContext()
	execCtx.AddEvent("user", model.Message{Role: "user", Content: userMessage})
	emit(ch, RunStartEvent{RunID: execCtx.id, UserMessage: userMessage})

	for execCtx.currentStep < a.maxSteps {
		if err := a.step(ctx, execCtx, ch); err != nil {
			err = fmt.Errorf("agent step %d: %w", execCtx.currentStep, err)
			emit(ch, RunEndEvent{RunID: execCtx.id, Err: err})
			return nil, err
		}
		if execCtx.Done() {
			break
		}
		execCtx.IncrementStep()
	}

	if !execCtx.Done() {
		err := fmt.Errorf("%w after %d steps", ErrMaxStepsReached, a.maxSteps)
		emit(ch, RunEndEvent{RunID: execCtx.id, Err: err})
		return nil, err
	}

	output, ok := execCtx.FinalResult()
	if !ok {
		err := fmt.Errorf("agent: internal error — finalResult type is %T, expected string", execCtx.finalResult)
		emit(ch, RunEndEvent{RunID: execCtx.id, Err: err})
		return nil, err
	}

	result := &Result{
		Output:     output,
		ToolCalled: anyToolCalled(execCtx),
		Context:    execCtx,
	}
	emit(ch, RunEndEvent{RunID: execCtx.id, Result: result})
	return result, nil
}

// Step executes one Think → (optionally) Act cycle, mutating execCtx in place.
// It is exported so callers can drive the loop manually for checkpointing or
// human-in-the-loop interrupts. Use execCtx.Done() to check for a final answer.
// Note: events are not emitted when calling Step directly; use Run for observability.
func (a *Agent) Step(ctx context.Context, execCtx *ExecutionContext) error {
	return a.step(ctx, execCtx, nil)
}

func (a *Agent) step(ctx context.Context, execCtx *ExecutionContext, ch chan<- AgentEvent) error {
	emit(ch, StepStartEvent{RunID: execCtx.id, Step: execCtx.currentStep})

	resp, err := a.think(ctx, execCtx, ch)
	if err != nil {
		emit(ch, StepEndEvent{RunID: execCtx.id, Step: execCtx.currentStep, Err: err})
		return err
	}

	toolCalls := collectToolCalls(resp.Content)
	if len(toolCalls) == 0 {
		if msg := extractAssistantMessage(resp.Content); msg != "" {
			execCtx.AddEvent("agent", model.Message{Role: "assistant", Content: msg})
			execCtx.setFinalResult(msg)
		}
		emit(ch, StepEndEvent{RunID: execCtx.id, Step: execCtx.currentStep})
		return nil
	}

	err = a.act(ctx, execCtx, toolCalls, ch)
	emit(ch, StepEndEvent{RunID: execCtx.id, Step: execCtx.currentStep, Err: err})
	return err
}

// Think calls the LLM with the current execution context and returns its response.
// Note: events are not emitted when calling Think directly; use Run for observability.
func (a *Agent) Think(ctx context.Context, execCtx *ExecutionContext) (model.Response, error) {
	return a.think(ctx, execCtx, nil)
}

func (a *Agent) think(ctx context.Context, execCtx *ExecutionContext, ch chan<- AgentEvent) (model.Response, error) {
	req := a.prepareRequest(execCtx)
	start := time.Now()
	resp, err := a.llmClient.Generate(ctx, req)
	emit(ch, LLMCallEvent{RunID: execCtx.id, Step: execCtx.currentStep, Latency: time.Since(start), Err: err})
	return resp, err
}

// Act executes all requested tool calls via ToolExecutor and records the results.
// The agent's tool-call decision is appended as an "agent" event BEFORE execution,
// then tool results are appended as a "tools" event AFTER execution.
// Note: events are not emitted when calling Act directly; use Run for observability.
func (a *Agent) Act(ctx context.Context, execCtx *ExecutionContext, calls []model.ToolCall) error {
	return a.act(ctx, execCtx, calls, nil)
}

func (a *Agent) act(ctx context.Context, execCtx *ExecutionContext, calls []model.ToolCall, ch chan<- AgentEvent) error {
	callItems := make([]model.ContentItem, len(calls))
	for i, tc := range calls {
		callItems[i] = tc
	}
	// Record the tool-call decision before execution so the log is accurate
	// even if the executor is nil or returns an error.
	execCtx.AddEvent("agent", callItems...)

	if a.executor == nil {
		err := fmt.Errorf("agent: cannot execute tools — ToolExecutor is nil")
		emit(ch, ToolExecEvent{RunID: execCtx.id, Step: execCtx.currentStep, ToolNames: toolNames(calls), Err: err})
		return err
	}

	start := time.Now()
	results, err := a.executor.Execute(ctx, calls)
	emit(ch, ToolExecEvent{RunID: execCtx.id, Step: execCtx.currentStep, ToolNames: toolNames(calls), Latency: time.Since(start), Err: err})
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

// emit sends an AgentEvent to ch. If ch is nil, it is a no-op.
// It never blocks: the channel buffer is always sized to hold all events for a run.
func emit(ch chan<- AgentEvent, event AgentEvent) {
	if ch != nil {
		ch <- event
	}
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
