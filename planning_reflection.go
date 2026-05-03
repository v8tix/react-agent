package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/v8tix/react-agent/model"
)

const planningReflectionRevisionPrompt = "Do not answer yet. Revise your task list with create_tasks based on what you learned before answering."
const planningReflectionStagnationPrompt = "Do not revise the plan again yet. First reply with a short reflection on why the plan is stalling and what you will change."
const planningReflectionRevisionFromReflectionPrompt = "Thanks. Now revise your task list with create_tasks based on that reflection before answering."

// PlanningReflectionOption configures a PlanningReflectionTracker.
type PlanningReflectionOption func(*PlanningReflectionTracker)

// WithPlanningReflectionStagnationThreshold enables stagnation-aware reflection
// after repeated planning-only revisions without meaningful progress.
func WithPlanningReflectionStagnationThreshold(n int) PlanningReflectionOption {
	return func(t *PlanningReflectionTracker) {
		if n > 0 {
			t.stagnationThreshold = n
		}
	}
}

// PlanningReflectionTracker coordinates plan revision after either an early
// answer or repeated planning-only stagnation.
type PlanningReflectionTracker struct {
	mu                          sync.Mutex
	revisionNeeded              bool
	reflectionNeeded            bool
	requestedAtRevisions        int
	planningRevisionsSinceReset int
	stagnationThreshold         int
	reflections                 []string
}

// NewPlanningReflectionTracker creates a new unified planning/reflection tracker.
func NewPlanningReflectionTracker(options ...PlanningReflectionOption) *PlanningReflectionTracker {
	tracker := &PlanningReflectionTracker{}
	for _, option := range options {
		if option != nil {
			option(tracker)
		}
	}
	return tracker
}

// BeforeTool blocks continued tool churn while a reflection is required.
func (t *PlanningReflectionTracker) BeforeTool(
	_ context.Context,
	execCtx *ExecutionContext,
	call model.ToolCall,
) (*model.ToolResult, error) {
	if !t.NeedsReflection() {
		return nil, nil
	}
	QueueDeferredUserMessage(execCtx, planningReflectionStagnationPrompt)
	return &model.ToolResult{
		ID:      call.ID,
		Name:    call.Name,
		Status:  "blocked",
		Content: []string{"reflection required before continuing"},
	}, nil
}

// AfterTool records when planning stagnates and when a later plan revision
// resolves a previously requested revision.
func (t *PlanningReflectionTracker) AfterTool(
	ctx context.Context,
	execCtx *ExecutionContext,
	result model.ToolResult,
) (*model.ToolResult, error) {
	if execCtx == nil || result.Status != "success" {
		return nil, nil
	}

	toolName := PlanningToolDefinition().Name
	if result.Name != toolName {
		t.mu.Lock()
		t.planningRevisionsSinceReset = 0
		t.mu.Unlock()
		return nil, nil
	}

	revisions := countPlanningRevisions(execCtx.Events()) + 1

	t.mu.Lock()
	t.planningRevisionsSinceReset++
	resolve := t.revisionNeeded && revisions > t.requestedAtRevisions
	if resolve {
		t.revisionNeeded = false
		t.planningRevisionsSinceReset = 0
	}
	stagnation := t.stagnationThreshold > 0 &&
		!t.reflectionNeeded &&
		!t.revisionNeeded &&
		t.planningRevisionsSinceReset >= t.stagnationThreshold
	if stagnation {
		t.reflectionNeeded = true
	}
	t.mu.Unlock()

	if stagnation {
		t.emit(ctx, execCtx, PlanningReflectionEvent{
			RunID:   execCtx.id,
			Step:    execCtx.currentStep,
			Kind:    PlanningReflectionEventStagnationObserved,
			Content: strings.Join(result.Content, "\n"),
		})
	}
	if resolve {
		t.emit(ctx, execCtx, PlanningReflectionEvent{
			RunID:   execCtx.id,
			Step:    execCtx.currentStep,
			Kind:    PlanningReflectionEventRevisionResolved,
			Content: strings.Join(result.Content, "\n"),
		})
	}
	return nil, nil
}

// NeedsRevision reports whether the current plan still must be revised before a
// final answer can be accepted.
func (t *PlanningReflectionTracker) NeedsRevision() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.revisionNeeded
}

// NeedsReflection reports whether the agent must reflect before continuing.
func (t *PlanningReflectionTracker) NeedsReflection() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.reflectionNeeded
}

// LatestReflection returns the most recently recorded planning reflection text.
func (t *PlanningReflectionTracker) LatestReflection() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.reflections) == 0 {
		return ""
	}
	return t.reflections[len(t.reflections)-1]
}

func (t *PlanningReflectionTracker) requestRevision(ctx context.Context, execCtx *ExecutionContext, answer string, revisions int) {
	if execCtx == nil {
		return
	}

	t.mu.Lock()
	if t.revisionNeeded {
		t.mu.Unlock()
		return
	}
	t.revisionNeeded = true
	t.requestedAtRevisions = revisions
	t.reflections = append(t.reflections, answer)
	t.mu.Unlock()

	t.emit(ctx, execCtx, PlanningReflectionEvent{
		RunID:   execCtx.id,
		Step:    execCtx.currentStep,
		Kind:    PlanningReflectionEventInsufficientProgress,
		Content: answer,
	})
	t.emit(ctx, execCtx, PlanningReflectionEvent{
		RunID:   execCtx.id,
		Step:    execCtx.currentStep,
		Kind:    PlanningReflectionEventReflectionRecorded,
		Content: answer,
	})
	t.emit(ctx, execCtx, PlanningReflectionEvent{
		RunID:   execCtx.id,
		Step:    execCtx.currentStep,
		Kind:    PlanningReflectionEventRevisionNeeded,
		Content: answer,
	})
}

// RecordReflection stores explicit reflection after a stagnation block and then
// requires a revised plan before the final answer.
func (t *PlanningReflectionTracker) RecordReflection(
	ctx context.Context,
	execCtx *ExecutionContext,
	reflection string,
	revisions int,
) {
	if execCtx == nil {
		return
	}

	t.mu.Lock()
	t.reflections = append(t.reflections, reflection)
	t.reflectionNeeded = false
	t.revisionNeeded = true
	t.requestedAtRevisions = revisions
	t.mu.Unlock()

	t.emit(ctx, execCtx, PlanningReflectionEvent{
		RunID:   execCtx.id,
		Step:    execCtx.currentStep,
		Kind:    PlanningReflectionEventReflectionRecorded,
		Content: reflection,
	})
	t.emit(ctx, execCtx, PlanningReflectionEvent{
		RunID:   execCtx.id,
		Step:    execCtx.currentStep,
		Kind:    PlanningReflectionEventRevisionNeeded,
		Content: reflection,
	})
}

func (t *PlanningReflectionTracker) emit(ctx context.Context, execCtx *ExecutionContext, event PlanningReflectionEvent) {
	if execCtx == nil {
		return
	}
	if ch, ok := ctx.Value(planningEventChannelKey{}).(chan<- AgentEvent); ok {
		emit(ch, event)
	}
}

// PlanningReflectionPolicy enforces that the agent revises its plan after an
// early draft answer or a stagnation-triggered reflection before it can finalize.
type PlanningReflectionPolicy struct {
	source       planRevisionSource
	tracker      *PlanningReflectionTracker
	minRevisions int
}

// NewPlanningReflectionPolicy creates a unified planning/reflection policy.
func NewPlanningReflectionPolicy(
	source planRevisionSource,
	tracker *PlanningReflectionTracker,
	minRevisions int,
) *PlanningReflectionPolicy {
	if minRevisions <= 0 {
		minRevisions = 2
	}
	return &PlanningReflectionPolicy{
		source:       source,
		tracker:      tracker,
		minRevisions: minRevisions,
	}
}

// BeforeFinalAnswer implements FinalAnswerCallback.
func (p *PlanningReflectionPolicy) BeforeFinalAnswer(
	ctx context.Context,
	execCtx *ExecutionContext,
	answer string,
) error {
	revisions := 0
	if p.source != nil {
		revisions = len(p.source.Revisions())
	}
	if p.tracker != nil && p.tracker.NeedsReflection() {
		p.tracker.RecordReflection(ctx, execCtx, answer, revisions)
		return fmt.Errorf(planningReflectionRevisionFromReflectionPrompt)
	}
	if revisions >= p.minRevisions && (p.tracker == nil || !p.tracker.NeedsRevision()) {
		return nil
	}
	if p.tracker != nil {
		p.tracker.requestRevision(ctx, execCtx, answer, revisions)
	}
	return fmt.Errorf(planningReflectionRevisionPrompt)
}

func countPlanningRevisions(events []model.Event) int {
	count := 0
	for _, event := range events {
		for _, item := range event.Content {
			result, ok := item.(model.ToolResult)
			if ok && result.Name == PlanningToolDefinition().Name && result.Status == "success" {
				count++
			}
		}
	}
	return count
}
