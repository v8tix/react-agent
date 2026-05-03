package agent

import (
	"context"
	"fmt"
	"sync"

	"github.com/v8tix/react-agent/model"
)

const verificationReflectionPrompt = "Do not gather more evidence yet. First reply with a short reflection that names the missing evidence and the next action."
const verificationRetryPrompt = "Thanks. Now gather the missing evidence before answering."

// EvidenceItem represents one supporting fact that can justify a final answer.
type EvidenceItem struct {
	Source  string
	Content string
	Score   float64
}

// EvidenceCollector exposes the current set of gathered evidence items.
type EvidenceCollector interface {
	Evidence() []EvidenceItem
}

// VerificationOption configures a VerificationGate.
type VerificationOption func(*VerificationGate)

// WithActionableVerificationReflection requires a short reflection before more
// evidence can be gathered after an insufficiently verified answer attempt.
func WithActionableVerificationReflection() VerificationOption {
	return func(g *VerificationGate) {
		g.reflectionPrompt = verificationReflectionPrompt
		g.retryPrompt = verificationRetryPrompt
	}
}

// EvidenceTracker records supporting evidence from successful tool results.
type EvidenceTracker struct {
	mu     sync.Mutex
	items  []EvidenceItem
	mapper func(model.ToolResult) (EvidenceItem, bool)
}

// NewEvidenceTracker creates a reusable evidence tracker with a caller-provided mapper.
func NewEvidenceTracker(mapper func(model.ToolResult) (EvidenceItem, bool)) *EvidenceTracker {
	return &EvidenceTracker{mapper: mapper}
}

// AfterTool implements AfterToolCallback.
func (t *EvidenceTracker) AfterTool(
	_ context.Context,
	_ *ExecutionContext,
	result model.ToolResult,
) (*model.ToolResult, error) {
	if t == nil || t.mapper == nil || result.Status != "success" {
		return nil, nil
	}
	item, ok := t.mapper(result)
	if !ok {
		return nil, nil
	}
	t.mu.Lock()
	t.items = append(t.items, item)
	t.mu.Unlock()
	return nil, nil
}

// Evidence returns the recorded evidence items.
func (t *EvidenceTracker) Evidence() []EvidenceItem {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]EvidenceItem(nil), t.items...)
}

// VerificationGate blocks final answers until enough evidence has been gathered.
type VerificationGate struct {
	collector        EvidenceCollector
	minItems         int
	reflectionPrompt string
	retryPrompt      string

	mu                  sync.Mutex
	reflectionNeeded    bool
	recordedReflections []string
}

// NewVerificationGate creates a reusable evidence gate.
func NewVerificationGate(collector EvidenceCollector, minItems int, options ...VerificationOption) *VerificationGate {
	if minItems <= 0 {
		minItems = 1
	}
	gate := &VerificationGate{
		collector: collector,
		minItems:  minItems,
	}
	for _, option := range options {
		if option != nil {
			option(gate)
		}
	}
	return gate
}

// BeforeTool blocks further evidence gathering until an actionable reflection is recorded.
func (g *VerificationGate) BeforeTool(
	_ context.Context,
	execCtx *ExecutionContext,
	call model.ToolCall,
) (*model.ToolResult, error) {
	if g == nil || !g.NeedsReflection() || g.reflectionPrompt == "" {
		return nil, nil
	}
	QueueDeferredUserMessage(execCtx, g.reflectionPrompt)
	return &model.ToolResult{
		ID:      call.ID,
		Name:    call.Name,
		Status:  "blocked",
		Content: []string{"reflection required before gathering more evidence"},
	}, nil
}

// BeforeFinalAnswer implements FinalAnswerCallback.
func (g *VerificationGate) BeforeFinalAnswer(
	_ context.Context,
	_ *ExecutionContext,
	answer string,
) error {
	if g == nil || g.collector == nil || len(g.collector.Evidence()) >= g.minItems {
		return nil
	}
	if g.NeedsReflection() && g.reflectionPrompt != "" {
		g.mu.Lock()
		g.recordedReflections = append(g.recordedReflections, answer)
		g.reflectionNeeded = false
		retryPrompt := g.retryPrompt
		g.mu.Unlock()
		if retryPrompt == "" {
			retryPrompt = verificationRetryPrompt
		}
		return fmt.Errorf("%s", retryPrompt)
	}
	if g.reflectionPrompt != "" {
		g.mu.Lock()
		g.reflectionNeeded = true
		g.mu.Unlock()
	}
	return fmt.Errorf("Do not answer yet. Evidence is incomplete. Gather at least %d supporting results before answering.", g.minItems)
}

// NeedsReflection reports whether the gate is waiting for an actionable reflection.
func (g *VerificationGate) NeedsReflection() bool {
	if g == nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.reflectionNeeded
}

// LatestReflection returns the most recently recorded reflection, if any.
func (g *VerificationGate) LatestReflection() string {
	if g == nil {
		return ""
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.recordedReflections) == 0 {
		return ""
	}
	return g.recordedReflections[len(g.recordedReflections)-1]
}
