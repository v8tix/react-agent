package agent

import (
	"crypto/rand"
	"fmt"
	"sync"
	"time"

	"github.com/v8tix/react-agent/model"
)

// ExecutionContext is the central mutable state for one agent run.
// It records all Events across steps and holds the final result once the
// agent produces a terminal response.
// All public methods are safe for concurrent use.
type ExecutionContext struct {
	mu                  sync.Mutex
	id                  string
	currentStep         int
	state               map[string]any
	events              []model.Event
	finalResult         any
	pendingInteraction  *InteractionRequest
	interactionResponse map[string]InteractionResponse
	pendingAct          *actState
}

func newExecutionContext() *ExecutionContext {
	return &ExecutionContext{
		id:                  generateID(),
		state:               make(map[string]any),
		interactionResponse: make(map[string]InteractionResponse),
	}
}

// NewExecutionContextForTest exposes newExecutionContext for white-box unit tests.
func NewExecutionContextForTest() *ExecutionContext { return newExecutionContext() }

// ID returns the unique identifier for this execution. Safe for concurrent use.
func (ec *ExecutionContext) ID() string {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	return ec.id
}

// CurrentStep returns the current step index. Safe for concurrent use.
func (ec *ExecutionContext) CurrentStep() int {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	return ec.currentStep
}

// Done reports whether the agent has produced a final answer. Safe for concurrent use.
func (ec *ExecutionContext) Done() bool {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	return ec.finalResult != nil
}

// FinalResult returns the agent's final answer and true once Done() is true.
// Returns ("", false) if the agent has not finished yet. Safe for concurrent use.
func (ec *ExecutionContext) FinalResult() (string, bool) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	s, ok := ec.finalResult.(string)
	return s, ok
}

// GetState retrieves a value from the run-scoped key-value store. Safe for concurrent use.
func (ec *ExecutionContext) GetState(key string) (any, bool) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	v, ok := ec.state[key]
	return v, ok
}

// SetState stores a value in the run-scoped key-value store. Safe for concurrent use.
func (ec *ExecutionContext) SetState(key string, value any) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.state[key] = value
}

// AddEvent appends an event authored by author with the given content items.
// ID and Timestamp are generated automatically. Safe for concurrent use.
func (ec *ExecutionContext) AddEvent(author string, content ...model.ContentItem) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.events = append(ec.events, model.Event{
		ID:          generateID(),
		ExecutionID: ec.id,
		Timestamp:   time.Now(),
		Author:      author,
		Content:     content,
	})
}

// Events returns a defensive copy of the event log.
// Each Event's Content slice is independently copied so callers cannot corrupt
// internal state by mutating returned slices. Safe for concurrent use.
func (ec *ExecutionContext) Events() []model.Event {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	out := make([]model.Event, len(ec.events))
	for i, e := range ec.events {
		out[i] = e
		if len(e.Content) > 0 {
			out[i].Content = make([]model.ContentItem, len(e.Content))
			copy(out[i].Content, e.Content)
		}
	}
	return out
}

// IncrementStep advances the step counter by one. Safe for concurrent use.
func (ec *ExecutionContext) IncrementStep() {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.currentStep++
}

// setFinalResult stores the terminal answer. Package-internal use only.
func (ec *ExecutionContext) setFinalResult(v any) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.finalResult = v
}

func (ec *ExecutionContext) setPendingInteraction(req InteractionRequest) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	reqCopy := req
	ec.pendingInteraction = &reqCopy
}

// PendingInteraction returns the active external interaction request, if any.
func (ec *ExecutionContext) PendingInteraction() (*InteractionRequest, bool) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	if ec.pendingInteraction == nil {
		return nil, false
	}
	req := *ec.pendingInteraction
	return &req, true
}

func (ec *ExecutionContext) clearPendingInteraction() {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.pendingInteraction = nil
}

func (ec *ExecutionContext) storeInteractionResponse(resp InteractionResponse) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.interactionResponse[resp.RequestID] = resp
}

// InteractionResponse returns a previously supplied external response, if present.
func (ec *ExecutionContext) InteractionResponse(requestID string) (InteractionResponse, bool) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	resp, ok := ec.interactionResponse[requestID]
	return resp, ok
}

func (ec *ExecutionContext) setPendingAct(state *actState) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.pendingAct = state
}

func (ec *ExecutionContext) pendingActState() (*actState, bool) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	if ec.pendingAct == nil {
		return nil, false
	}
	return ec.pendingAct, true
}

func (ec *ExecutionContext) clearPendingAct() {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.pendingAct = nil
}

func generateID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)
}
