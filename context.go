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
	mu          sync.Mutex
	ID          string
	CurrentStep int
	State       map[string]any
	events      []model.Event
	finalResult any
}

func newExecutionContext() *ExecutionContext {
	return &ExecutionContext{
		ID:    generateID(),
		State: make(map[string]any),
	}
}

// NewExecutionContextForTest exposes newExecutionContext for white-box unit tests.
func NewExecutionContextForTest() *ExecutionContext { return newExecutionContext() }

// AddEvent appends an event authored by author with the given content items.
// ID and Timestamp are generated automatically. Safe for concurrent use.
func (ec *ExecutionContext) AddEvent(author string, content ...model.ContentItem) {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.events = append(ec.events, model.Event{
		ID:          generateID(),
		ExecutionID: ec.ID,
		Timestamp:   time.Now(),
		Author:      author,
		Content:     content,
	})
}

// Events returns a defensive copy of the event log. Safe for concurrent use.
func (ec *ExecutionContext) Events() []model.Event {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	out := make([]model.Event, len(ec.events))
	copy(out, ec.events)
	return out
}

// IncrementStep advances the step counter by one. Safe for concurrent use.
func (ec *ExecutionContext) IncrementStep() {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	ec.CurrentStep++
}

func generateID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)
}
