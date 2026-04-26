package agent

import (
	"time"

	"github.com/v8tix/react-agent/model"
)

// AgentEvent is the sealed sum type for all agent lifecycle events.
// Use a type switch to handle specific event types:
//
//	for item := range events.Observe() {
//	    switch e := item.V.(type) {
//	    case agent.LLMCallEvent:
//	        slog.Info("llm call", "latency_ms", e.Latency.Milliseconds())
//	    case agent.RunEndEvent:
//	        fmt.Println(e.Result.Output)
//	    }
//	}
type AgentEvent interface{ isAgentEvent() }

// RunStartEvent is emitted once before the ReAct loop begins.
type RunStartEvent struct {
	RunID       string
	UserMessage string
}

// RunEndEvent is emitted once when Run returns, whether it succeeded.
// Result is nil when Err is non-nil.
type RunEndEvent struct {
	RunID  string
	Result *Result
	Err    error
}

// StepStartEvent is emitted at the beginning of each Think→Act cycle.
type StepStartEvent struct {
	RunID string
	Step  int
}

// StepEndEvent is emitted after each Think→Act cycle.
// Err is non-nil when the step failed.
type StepEndEvent struct {
	RunID string
	Step  int
	Err   error
}

// LLMCallEvent is emitted after every Generate() call, including on error.
// Latency covers only the LLM network round-trip.
// The full request and response content are available via result.Context.Events().
type LLMCallEvent struct {
	RunID   string
	Step    int
	Latency time.Duration
	Err     error
}

// ToolExecEvent is emitted after every executor.Execute() batch.
// ToolNames lists the names of tools that were called.
// Latency is 0 when the executor is nil.
type ToolExecEvent struct {
	RunID     string
	Step      int
	ToolNames []string
	Latency   time.Duration
	Err       error
}

// CallbackPhase identifies which callback stage emitted the event.
type CallbackPhase string

const (
	CallbackPhaseBeforeTool CallbackPhase = "before_tool"
	CallbackPhaseAfterTool  CallbackPhase = "after_tool"
)

// CallbackStage identifies whether the event was emitted before invoking the
// callback or after it returned.
type CallbackStage string

const (
	CallbackStageStart  CallbackStage = "start"
	CallbackStageFinish CallbackStage = "finish"
)

// CallbackEvent is emitted before and after each callback invocation.
type CallbackEvent struct {
	RunID      string
	Step       int
	Phase      CallbackPhase
	Stage      CallbackStage
	Callback   string
	ToolCallID string
	ToolName   string
	Overrode   bool
	Latency    time.Duration
	Err        error
}

// InteractionRequestedEvent is emitted when the agent suspends to await external input.
type InteractionRequestedEvent struct {
	RunID   string
	Step    int
	Request InteractionRequest
}

// InteractionResumedEvent is emitted when a suspended interaction receives a response.
type InteractionResumedEvent struct {
	RunID    string
	Step     int
	Response InteractionResponse
}

func (RunStartEvent) isAgentEvent()             {}
func (RunEndEvent) isAgentEvent()               {}
func (StepStartEvent) isAgentEvent()            {}
func (StepEndEvent) isAgentEvent()              {}
func (LLMCallEvent) isAgentEvent()              {}
func (ToolExecEvent) isAgentEvent()             {}
func (CallbackEvent) isAgentEvent()             {}
func (InteractionRequestedEvent) isAgentEvent() {}
func (InteractionResumedEvent) isAgentEvent()   {}

func toolNames(calls []model.ToolCall) []string {
	names := make([]string, len(calls))
	for i, c := range calls {
		names[i] = c.Name
	}
	return names
}
