package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/reactivex/rxgo/v2"
	"github.com/v8tix/react-agent/model"
)

// Agent is the ReAct orchestrator. It runs a Think → Act → Observe loop until
// the LLM produces a final answer or maxSteps is exhausted.
type Agent struct {
	llmClient           LLMClient
	toolDefs            []model.ToolDefinition // definitions sent to the LLM each turn
	executor            model.ToolExecutor     // nil is safe when toolDefs is empty
	instructions        string
	maxSteps            int
	beforeToolCallbacks []BeforeToolCallback
	afterToolCallbacks  []AfterToolCallback
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

// WithBeforeToolCallbacks appends tool callbacks that run before executor dispatch.
func (a *Agent) WithBeforeToolCallbacks(callbacks ...BeforeToolCallback) *Agent {
	a.beforeToolCallbacks = append(a.beforeToolCallbacks, callbacks...)
	return a
}

// WithAfterToolCallbacks appends tool callbacks that run after a tool result is produced.
func (a *Agent) WithAfterToolCallbacks(callbacks ...AfterToolCallback) *Agent {
	a.afterToolCallbacks = append(a.afterToolCallbacks, callbacks...)
	return a
}

// Run executes the full ReAct loop for a single user message.
// It returns a Result, a replayable Observable of AgentEvents, and any error.
//
// The Observable is a cold, replayable stream: each call to Observe() replays
// all events from the completed run. It is safe for multiple subscribers.
func (a *Agent) Run(ctx context.Context, userMessage string) (*Result, rxgo.Observable, error) {
	ch := make(chan AgentEvent, a.maxSteps*8+16)
	result, err := a.runCore(ctx, userMessage, ch)
	return result, observableFromChannel(ch), err
}

// runCore is the internal loop. It writes AgentEvents to ch (never closes it).
func (a *Agent) runCore(ctx context.Context, userMessage string, ch chan<- AgentEvent) (*Result, error) {
	execCtx := newExecutionContext()
	execCtx.AddEvent("user", model.Message{Role: "user", Content: userMessage})
	emit(ch, RunStartEvent{RunID: execCtx.id, UserMessage: userMessage})
	return a.continueRun(ctx, execCtx, ch, false)
}

// Resume continues a suspended run after an external interaction response arrives.
func (a *Agent) Resume(ctx context.Context, suspended SuspendedRun, response InteractionResponse) (*Result, rxgo.Observable, error) {
	ch := make(chan AgentEvent, a.maxSteps*8+16)
	execCtx := suspended.Context
	pending, ok := execCtx.PendingInteraction()
	if !ok {
		err := fmt.Errorf("agent: resume requested without pending interaction")
		emit(ch, RunEndEvent{RunID: execCtx.id, Err: err})
		return nil, observableFromChannel(ch), err
	}
	if pending.ID != response.RequestID {
		err := fmt.Errorf("agent: resume request id mismatch: want %s, got %s", pending.ID, response.RequestID)
		emit(ch, RunEndEvent{RunID: execCtx.id, Err: err})
		return nil, observableFromChannel(ch), err
	}
	execCtx.storeInteractionResponse(response)
	execCtx.clearPendingInteraction()
	emit(ch, InteractionResumedEvent{RunID: execCtx.id, Step: execCtx.currentStep, Response: response})
	result, err := a.continueRun(ctx, execCtx, ch, true)
	return result, observableFromChannel(ch), err
}

func (a *Agent) continueRun(ctx context.Context, execCtx *ExecutionContext, ch chan<- AgentEvent, resumePendingAct bool) (*Result, error) {
	if resumePendingAct {
		err := a.resumeAct(ctx, execCtx, ch)
		emit(ch, StepEndEvent{RunID: execCtx.id, Step: execCtx.currentStep, Err: err})
		if err != nil {
			err = fmt.Errorf("agent step %d: %w", execCtx.currentStep, err)
			emit(ch, RunEndEvent{RunID: execCtx.id, Err: err})
			return nil, err
		}
		if !execCtx.Done() {
			execCtx.IncrementStep()
		}
	}

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

func observableFromChannel(ch chan AgentEvent) rxgo.Observable {
	close(ch)
	collected := make([]AgentEvent, 0, len(ch))
	for e := range ch {
		collected = append(collected, e)
	}
	return rxgo.Defer([]rxgo.Producer{
		func(_ context.Context, out chan<- rxgo.Item) {
			for _, e := range collected {
				out <- rxgo.Of(e)
			}
		},
	})
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
	execCtx.AddEvent("agent", callItems...)
	execCtx.setPendingAct(&actState{
		Calls:   calls,
		Results: make([]model.ToolResult, len(calls)),
		Phase:   CallbackPhaseBeforeTool,
	})
	return a.resumeAct(ctx, execCtx, ch)
}

func (a *Agent) resumeAct(ctx context.Context, execCtx *ExecutionContext, ch chan<- AgentEvent) error {
	state, ok := execCtx.pendingActState()
	if !ok {
		return nil
	}

	for state.Phase == CallbackPhaseBeforeTool {
		for state.CurrentCall < len(state.Calls) {
			call := state.Calls[state.CurrentCall]
			override, suspended, err := a.resumeBeforeCallbacks(ctx, execCtx, state, call, ch)
			if suspended || err != nil {
				if err != nil && !errors.Is(err, ErrInteractionRequested) {
					execCtx.clearPendingAct()
				}
				return err
			}
			if override != nil {
				state.Results[state.CurrentCall] = *override
				state.CurrentCall++
				state.CurrentCallback = 0
				continue
			}
			state.PendingCalls = append(state.PendingCalls, call)
			state.PendingIndices = append(state.PendingIndices, state.CurrentCall)
			state.CurrentCall++
			state.CurrentCallback = 0
		}
		state.Phase = CallbackPhaseAfterTool
		state.CurrentCall = 0
		state.CurrentCallback = 0
	}

	if !state.ExecutorDone {
		if len(state.PendingCalls) > 0 {
			if a.executor == nil {
				err := fmt.Errorf("agent: cannot execute tools — ToolExecutor is nil")
				emit(ch, ToolExecEvent{RunID: execCtx.id, Step: execCtx.currentStep, ToolNames: toolNames(state.PendingCalls), Err: err})
				execCtx.clearPendingAct()
				return err
			}
			start := time.Now()
			executedResults, err := a.executor.Execute(ctx, state.PendingCalls)
			emit(ch, ToolExecEvent{RunID: execCtx.id, Step: execCtx.currentStep, ToolNames: toolNames(state.PendingCalls), Latency: time.Since(start), Err: err})
			if err != nil {
				execCtx.clearPendingAct()
				return fmt.Errorf("act execute: %w", err)
			}
			if len(executedResults) != len(state.PendingCalls) {
				execCtx.clearPendingAct()
				return fmt.Errorf("act execute: executor returned %d results for %d calls", len(executedResults), len(state.PendingCalls))
			}
			for i, result := range executedResults {
				state.Results[state.PendingIndices[i]] = normalizeToolResult(state.PendingCalls[i], result)
			}
		}
		state.ExecutorDone = true
	}

	for state.CurrentCall < len(state.Calls) {
		call := state.Calls[state.CurrentCall]
		result, suspended, err := a.resumeAfterCallbacks(ctx, execCtx, state, call, state.Results[state.CurrentCall], ch)
		if suspended || err != nil {
			if err != nil && !errors.Is(err, ErrInteractionRequested) {
				execCtx.clearPendingAct()
			}
			return err
		}
		state.Results[state.CurrentCall] = result
		state.CurrentCall++
		state.CurrentCallback = 0
	}

	resultItems := make([]model.ContentItem, len(state.Results))
	for i, tr := range state.Results {
		resultItems[i] = tr
	}
	execCtx.AddEvent("tools", resultItems...)
	execCtx.clearPendingAct()
	return nil
}

func callbackName(callback any) string {
	t := reflect.TypeOf(callback)
	if t == nil {
		return ""
	}
	return t.String()
}

func (a *Agent) resumeBeforeCallbacks(ctx context.Context, execCtx *ExecutionContext, state *actState, call model.ToolCall, ch chan<- AgentEvent) (*model.ToolResult, bool, error) {
	for state.CurrentCallback < len(a.beforeToolCallbacks) {
		callback := a.beforeToolCallbacks[state.CurrentCallback]
		emit(ch, CallbackEvent{
			RunID:      execCtx.id,
			Step:       execCtx.currentStep,
			Phase:      CallbackPhaseBeforeTool,
			Stage:      CallbackStageStart,
			Callback:   callbackName(callback),
			ToolCallID: call.ID,
			ToolName:   call.Name,
		})
		start := time.Now()
		override, err := callback.BeforeTool(ctx, execCtx, call)
		if sig, ok := asInteractionSignal(err); ok {
			req := normalizeInteractionRequest(sig.Request, call)
			execCtx.setPendingInteraction(req)
			emit(ch, CallbackEvent{
				RunID:      execCtx.id,
				Step:       execCtx.currentStep,
				Phase:      CallbackPhaseBeforeTool,
				Stage:      CallbackStageFinish,
				Callback:   callbackName(callback),
				ToolCallID: call.ID,
				ToolName:   call.Name,
				Overrode:   false,
				Latency:    time.Since(start),
			})
			emit(ch, InteractionRequestedEvent{RunID: execCtx.id, Step: execCtx.currentStep, Request: req})
			return nil, true, &InteractionRequestedError{Suspended: SuspendedRun{Context: execCtx, Interaction: req}}
		}
		emit(ch, CallbackEvent{
			RunID:      execCtx.id,
			Step:       execCtx.currentStep,
			Phase:      CallbackPhaseBeforeTool,
			Stage:      CallbackStageFinish,
			Callback:   callbackName(callback),
			ToolCallID: call.ID,
			ToolName:   call.Name,
			Overrode:   override != nil,
			Latency:    time.Since(start),
			Err:        err,
		})
		if err != nil {
			return nil, false, fmt.Errorf("act before-tool callback: %w", err)
		}
		state.CurrentCallback++
		if override != nil {
			normalized := normalizeToolResult(call, *override)
			return &normalized, false, nil
		}
	}
	return nil, false, nil
}

func (a *Agent) resumeAfterCallbacks(ctx context.Context, execCtx *ExecutionContext, state *actState, call model.ToolCall, result model.ToolResult, ch chan<- AgentEvent) (model.ToolResult, bool, error) {
	current := normalizeToolResult(call, result)
	for state.CurrentCallback < len(a.afterToolCallbacks) {
		callback := a.afterToolCallbacks[state.CurrentCallback]
		emit(ch, CallbackEvent{
			RunID:      execCtx.id,
			Step:       execCtx.currentStep,
			Phase:      CallbackPhaseAfterTool,
			Stage:      CallbackStageStart,
			Callback:   callbackName(callback),
			ToolCallID: call.ID,
			ToolName:   call.Name,
		})
		start := time.Now()
		override, err := callback.AfterTool(ctx, execCtx, current)
		if sig, ok := asInteractionSignal(err); ok {
			req := normalizeInteractionRequest(sig.Request, call)
			execCtx.setPendingInteraction(req)
			emit(ch, CallbackEvent{
				RunID:      execCtx.id,
				Step:       execCtx.currentStep,
				Phase:      CallbackPhaseAfterTool,
				Stage:      CallbackStageFinish,
				Callback:   callbackName(callback),
				ToolCallID: call.ID,
				ToolName:   call.Name,
				Overrode:   false,
				Latency:    time.Since(start),
			})
			emit(ch, InteractionRequestedEvent{RunID: execCtx.id, Step: execCtx.currentStep, Request: req})
			return model.ToolResult{}, true, &InteractionRequestedError{Suspended: SuspendedRun{Context: execCtx, Interaction: req}}
		}
		emit(ch, CallbackEvent{
			RunID:      execCtx.id,
			Step:       execCtx.currentStep,
			Phase:      CallbackPhaseAfterTool,
			Stage:      CallbackStageFinish,
			Callback:   callbackName(callback),
			ToolCallID: call.ID,
			ToolName:   call.Name,
			Overrode:   override != nil,
			Latency:    time.Since(start),
			Err:        err,
		})
		if err != nil {
			return model.ToolResult{}, false, fmt.Errorf("act after-tool callback: %w", err)
		}
		state.CurrentCallback++
		if override != nil {
			return normalizeToolResult(call, *override), false, nil
		}
	}
	return current, false, nil
}

func normalizeInteractionRequest(req InteractionRequest, call model.ToolCall) InteractionRequest {
	if req.ID == "" {
		req.ID = "interaction-" + call.ID
	}
	if req.ToolCallID == "" {
		req.ToolCallID = call.ID
	}
	if req.ToolName == "" {
		req.ToolName = call.Name
	}
	return req
}

func normalizeToolResult(call model.ToolCall, result model.ToolResult) model.ToolResult {
	if result.ID == "" {
		result.ID = call.ID
	}
	if result.Name == "" {
		result.Name = call.Name
	}
	if result.Status == "" {
		result.Status = "success"
	}
	return result
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
