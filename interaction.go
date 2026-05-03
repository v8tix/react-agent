package agent

import (
	"errors"
	"fmt"

	"github.com/v8tix/react-agent/model"
)

// ErrInteractionRequested signals that the agent suspended awaiting external input.
var ErrInteractionRequested = errors.New("agent: interaction requested")

// InteractionRequest describes a question that must be answered from outside
// the agent before the run can continue.
//
// Common examples are "approve this tool call?" or "which account should be
// used for this action?"
type InteractionRequest struct {
	ID         string
	Kind       string
	Prompt     string
	ToolCallID string
	ToolName   string
	Payload    map[string]any
}

// InteractionResponse carries the external answer to a pending interaction
// request.
type InteractionResponse struct {
	RequestID string
	Approved  *bool
	Value     string
	Metadata  map[string]any
}

// SuspendedRun contains the paused execution state plus the interaction that
// must be answered before the run can resume.
type SuspendedRun struct {
	Context     *ExecutionContext
	Interaction InteractionRequest
}

// InteractionRequestedError exposes a suspended run while still matching ErrInteractionRequested.
type InteractionRequestedError struct {
	Suspended SuspendedRun
}

func (e *InteractionRequestedError) Error() string {
	return fmt.Sprintf("%s: %s", ErrInteractionRequested.Error(), e.Suspended.Interaction.ID)
}

func (e *InteractionRequestedError) Unwrap() error { return ErrInteractionRequested }

type interactionSignal struct {
	Request InteractionRequest
}

func (e *interactionSignal) Error() string {
	return fmt.Sprintf("interaction requested: %s", e.Request.ID)
}

// Suspend requests external interaction from inside a callback.
//
// Returning Suspend(req) from a callback is what turns a normal run into an
// approval loop or any other human-in-the-loop step.
func Suspend(req InteractionRequest) error {
	return &interactionSignal{Request: req}
}

func asInteractionSignal(err error) (*interactionSignal, bool) {
	if sig, ok := errors.AsType[*interactionSignal](err); ok {
		return sig, true
	}
	return nil, false
}

type actState struct {
	Calls           []model.ToolCall
	Results         []model.ToolResult
	PendingCalls    []model.ToolCall
	PendingIndices  []int
	CurrentCall     int
	CurrentCallback int
	Phase           CallbackPhase
	ExecutorDone    bool
}
