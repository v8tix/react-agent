package agent

import (
	"time"

	"github.com/v8tix/react-agent/model"
)

// AgentEvent is the sealed sum type for all agent lifecycle events.
// Events are collected during a run and replayed afterward through a cold
// observable, so multiple Observe() calls are safe and deterministic.
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

// LiveEventSink consumes agent events as they are emitted during a run.
//
// Unlike the replayable observable returned by [Agent.Run], a live sink receives
// events while the run is still in progress. Sinks should stay lightweight and
// non-blocking because they execute on the event collector goroutine.
type LiveEventSink func(AgentEvent)

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

// PolicyDecision is the outcome of a final-answer policy check.
type PolicyDecision string

const (
	PolicyDecisionAccept PolicyDecision = "accept"
	PolicyDecisionReject PolicyDecision = "reject"
)

// PolicyEvent is emitted after a final-answer callback evaluates a proposed
// answer. It makes policy decisions observable in the same stream as other
// agent lifecycle events.
type PolicyEvent struct {
	RunID      string
	Step       int
	PolicyName string
	Decision   PolicyDecision
	Answer     string
	Reason     string
	Latency    time.Duration
}

// PlanRevisionEvent is emitted when a planning tool call records a new
// revision.
type PlanRevisionEvent struct {
	RunID    string
	Step     int
	Revision PlanRevision
}

// RecoveryEventKind describes where the agent is in an error-recovery flow.
type RecoveryEventKind string

const (
	RecoveryEventFailureObserved    RecoveryEventKind = "failure_observed"
	RecoveryEventRecovered          RecoveryEventKind = "recovered"
	RecoveryEventReflectionRecorded RecoveryEventKind = "reflection_recorded"
)

// RecoveryEvent is emitted when a recovery tracker observes a failed tool
// result or a successful retry.
type RecoveryEvent struct {
	RunID      string
	Step       int
	Kind       RecoveryEventKind
	ToolCallID string
	ToolName   string
	Reason     string
}

// SynthesisEventKind describes the type of synthesis event.
type SynthesisEventKind string

const (
	SynthesisEventObservationRecorded SynthesisEventKind = "observation_recorded"
	SynthesisEventSynthesisComplete   SynthesisEventKind = "synthesis_complete"
)

// SynthesisEvent is emitted when a synthesis tracker records an observation
// or completes a synthesis.
type SynthesisEvent struct {
	RunID      string
	Step       int
	Kind       SynthesisEventKind
	ToolCallID string
	ToolName   string
	Content    string
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

// PlanningReflectionEventKind describes the type of planning/reflection event.
type PlanningReflectionEventKind string

const (
	PlanningReflectionEventInsufficientProgress PlanningReflectionEventKind = "insufficient_progress"
	PlanningReflectionEventStagnationObserved   PlanningReflectionEventKind = "stagnation_observed"
	PlanningReflectionEventReflectionRecorded   PlanningReflectionEventKind = "reflection_recorded"
	PlanningReflectionEventRevisionNeeded       PlanningReflectionEventKind = "revision_needed"
	PlanningReflectionEventRevisionResolved     PlanningReflectionEventKind = "revision_resolved"
)

// PlanningReflectionEvent is emitted when the unified planning/reflection tracker
// detects insufficient progress or records a reflection about plan revision needs.
type PlanningReflectionEvent struct {
	RunID   string
	Step    int
	Kind    PlanningReflectionEventKind
	Content string
}

func (RunStartEvent) isAgentEvent()             {}
func (RunEndEvent) isAgentEvent()               {}
func (StepStartEvent) isAgentEvent()            {}
func (StepEndEvent) isAgentEvent()              {}
func (LLMCallEvent) isAgentEvent()              {}
func (ToolExecEvent) isAgentEvent()             {}
func (CallbackEvent) isAgentEvent()             {}
func (PolicyEvent) isAgentEvent()               {}
func (PlanRevisionEvent) isAgentEvent()         {}
func (RecoveryEvent) isAgentEvent()             {}
func (SynthesisEvent) isAgentEvent()            {}
func (InteractionRequestedEvent) isAgentEvent() {}
func (InteractionResumedEvent) isAgentEvent()   {}
func (PlanningReflectionEvent) isAgentEvent()   {}

func toolNames(calls []model.ToolCall) []string {
	names := make([]string, len(calls))
	for i, c := range calls {
		names[i] = c.Name
	}
	return names
}
