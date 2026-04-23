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

func (RunStartEvent) isAgentEvent()  {}
func (RunEndEvent) isAgentEvent()    {}
func (StepStartEvent) isAgentEvent() {}
func (StepEndEvent) isAgentEvent()   {}
func (LLMCallEvent) isAgentEvent()   {}
func (ToolExecEvent) isAgentEvent()  {}

func toolNames(calls []model.ToolCall) []string {
	names := make([]string, len(calls))
	for i, c := range calls {
		names[i] = c.Name
	}
	return names
}
