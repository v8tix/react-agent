package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/v8tix/react-agent/model"
)

const recoveryReflectionPrompt = "Do not call any tool yet. First reply with a short reflection on what failed and how you will correct the retry."

// RecoveryFailure captures a failed tool result that may require reflection and
// a retry before the agent can finish.
type RecoveryFailure struct {
	ToolCallID string
	ToolName   string
	Reason     string
}

// RecoveryAttempt captures a successful retry after a previous failure.
type RecoveryAttempt struct {
	ToolCallID string
	ToolName   string
	Output     string
}

type recoveryStateSource interface {
	HasUnresolvedFailures() bool
	RequiresReflection() bool
	RecordReflection(context.Context, *ExecutionContext, string) error
}

// RecoveryTracker records failed tool results and successful retries. It can be
// plugged directly into the agent as an AfterToolCallback.
type RecoveryTracker struct {
	mu              sync.Mutex
	failures        []RecoveryFailure
	attempts        []RecoveryAttempt
	reflections     []string
	unresolved      map[string]int
	needsReflection bool
}

// NewRecoveryTracker creates a reusable recovery tracker.
func NewRecoveryTracker() *RecoveryTracker {
	return &RecoveryTracker{
		unresolved: make(map[string]int),
	}
}

// BeforeTool blocks retries until a reflection message has been recorded after a failure.
func (r *RecoveryTracker) BeforeTool(
	_ context.Context,
	execCtx *ExecutionContext,
	call model.ToolCall,
) (*model.ToolResult, error) {
	r.mu.Lock()
	block := len(r.unresolved) > 0 && r.needsReflection
	r.mu.Unlock()
	if !block {
		return nil, nil
	}
	QueueDeferredUserMessage(execCtx, recoveryReflectionPrompt)
	return &model.ToolResult{
		ID:      call.ID,
		Name:    call.Name,
		Status:  "blocked",
		Content: []string{"retry blocked: do not call any tool yet; first reply with a short reflection on what failed and how you will correct the retry"},
	}, nil
}

// AfterTool records failed tool results and successful recovery attempts.
func (r *RecoveryTracker) AfterTool(
	ctx context.Context,
	execCtx *ExecutionContext,
	result model.ToolResult,
) (*model.ToolResult, error) {
	switch result.Status {
	case "error":
		failure := RecoveryFailure{
			ToolCallID: result.ID,
			ToolName:   result.Name,
			Reason:     strings.Join(result.Content, "\n"),
		}
		r.mu.Lock()
		r.failures = append(r.failures, failure)
		r.unresolved[result.Name]++
		r.needsReflection = true
		r.mu.Unlock()
		r.emit(ctx, execCtx, RecoveryEvent{
			RunID:      execCtx.id,
			Step:       execCtx.currentStep,
			Kind:       RecoveryEventFailureObserved,
			ToolCallID: result.ID,
			ToolName:   result.Name,
			Reason:     failure.Reason,
		})
	case "success":
		r.mu.Lock()
		if r.unresolved[result.Name] > 0 {
			r.unresolved[result.Name]--
			if r.unresolved[result.Name] == 0 {
				delete(r.unresolved, result.Name)
			}
			attempt := RecoveryAttempt{
				ToolCallID: result.ID,
				ToolName:   result.Name,
				Output:     strings.Join(result.Content, "\n"),
			}
			r.attempts = append(r.attempts, attempt)
			r.mu.Unlock()
			r.emit(ctx, execCtx, RecoveryEvent{
				RunID:      execCtx.id,
				Step:       execCtx.currentStep,
				Kind:       RecoveryEventRecovered,
				ToolCallID: result.ID,
				ToolName:   result.Name,
				Reason:     attempt.Output,
			})
			return nil, nil
		}
		r.mu.Unlock()
	}
	return nil, nil
}

// Failures returns the recorded failures.
func (r *RecoveryTracker) Failures() []RecoveryFailure {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]RecoveryFailure(nil), r.failures...)
}

// Attempts returns the recorded successful recovery attempts.
func (r *RecoveryTracker) Attempts() []RecoveryAttempt {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]RecoveryAttempt(nil), r.attempts...)
}

// HasUnresolvedFailures reports whether any tool failures still lack a
// successful follow-up attempt.
func (r *RecoveryTracker) HasUnresolvedFailures() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.unresolved) > 0
}

// RequiresReflection reports whether a reflection message is still required before retrying.
func (r *RecoveryTracker) RequiresReflection() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.needsReflection
}

// RecordReflection stores the recovery reflection text and clears the reflection requirement.
func (r *RecoveryTracker) RecordReflection(
	ctx context.Context,
	execCtx *ExecutionContext,
	reflection string,
) error {
	r.mu.Lock()
	if len(r.unresolved) == 0 || !r.needsReflection || strings.TrimSpace(reflection) == "" {
		r.mu.Unlock()
		return nil
	}
	r.reflections = append(r.reflections, reflection)
	r.needsReflection = false
	r.mu.Unlock()
	r.emit(ctx, execCtx, RecoveryEvent{
		RunID:  execCtx.id,
		Step:   execCtx.currentStep,
		Kind:   RecoveryEventReflectionRecorded,
		Reason: reflection,
	})
	return nil
}

// LatestReflection returns the most recently recorded reflection text.
func (r *RecoveryTracker) LatestReflection() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.reflections) == 0 {
		return ""
	}
	return r.reflections[len(r.reflections)-1]
}

func (r *RecoveryTracker) emit(ctx context.Context, execCtx *ExecutionContext, event RecoveryEvent) {
	if execCtx == nil {
		return
	}
	ch, _ := ctx.Value(planningEventChannelKey{}).(chan<- AgentEvent)
	emit(ch, event)
}

// RecoveryPolicy blocks final answers while unresolved tool failures remain.
type RecoveryPolicy struct {
	source recoveryStateSource
}

// NewRecoveryPolicy creates a reusable recovery policy.
func NewRecoveryPolicy(source recoveryStateSource) *RecoveryPolicy {
	return &RecoveryPolicy{source: source}
}

// BeforeFinalAnswer implements FinalAnswerCallback.
func (p *RecoveryPolicy) BeforeFinalAnswer(
	ctx context.Context,
	execCtx *ExecutionContext,
	answer string,
) error {
	if p.source == nil || !p.source.HasUnresolvedFailures() {
		return nil
	}
	if p.source.RequiresReflection() {
		if err := p.source.RecordReflection(ctx, execCtx, answer); err != nil {
			return err
		}
		return fmt.Errorf("Thanks. Now retry the corrected approach before answering.")
	}
	return fmt.Errorf("Do not answer yet. A tool failure is still unresolved. Retry the corrected approach before answering.")
}
