package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/v8tix/react-agent/model"
)

// Observation is a tool result captured during synthesis.
type Observation struct {
	ToolCallID string
	ToolName   string
	Content    string
}

// SynthesisRecord captures a completed synthesis with its observations.
type SynthesisRecord struct {
	Observations []Observation
}

type synthesisStateSource interface {
	HasIncompleteAnalysis() bool
	MarkSynthesisComplete(ctx context.Context, execCtx *ExecutionContext) error
}

// SynthesisTracker records tool observations and tracks synthesis completion.
// It can be plugged directly into the agent as an AfterToolCallback.
type SynthesisTracker struct {
	mu               sync.Mutex
	observations     []Observation
	synthesisHistory []SynthesisRecord
	incomplete       bool
}

// NewSynthesisTracker creates a reusable synthesis tracker.
func NewSynthesisTracker() *SynthesisTracker {
	return &SynthesisTracker{
		incomplete: false,
	}
}

// AfterTool records tool results as observations.
func (s *SynthesisTracker) AfterTool(
	ctx context.Context,
	execCtx *ExecutionContext,
	result model.ToolResult,
) (*model.ToolResult, error) {
	if result.Status == "success" {
		observation := Observation{
			ToolCallID: result.ID,
			ToolName:   result.Name,
			Content:    strings.Join(result.Content, "\n"),
		}
		s.mu.Lock()
		s.observations = append(s.observations, observation)
		s.incomplete = true
		s.mu.Unlock()
		s.emit(ctx, execCtx, SynthesisEvent{
			RunID:      execCtx.ID(),
			Step:       execCtx.CurrentStep(),
			Kind:       SynthesisEventObservationRecorded,
			ToolCallID: result.ID,
			ToolName:   result.Name,
			Content:    observation.Content,
		})
	}
	return nil, nil
}

// Observations returns the current observations.
func (s *SynthesisTracker) Observations() []Observation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Observation(nil), s.observations...)
}

// HasIncompleteAnalysis reports whether analysis remains incomplete.
func (s *SynthesisTracker) HasIncompleteAnalysis() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.incomplete
}

// MarkSynthesisComplete marks the current synthesis as complete and starts tracking next one.
func (s *SynthesisTracker) MarkSynthesisComplete(ctx context.Context, execCtx *ExecutionContext) error {
	if execCtx == nil {
		return nil
	}

	var event *SynthesisEvent
	s.mu.Lock()
	if s.incomplete {
		record := SynthesisRecord{
			Observations: append([]Observation(nil), s.observations...),
		}
		s.synthesisHistory = append(s.synthesisHistory, record)
		s.observations = nil
		s.incomplete = false

		event = &SynthesisEvent{
			RunID:   execCtx.ID(),
			Step:    execCtx.CurrentStep(),
			Kind:    SynthesisEventSynthesisComplete,
			Content: fmt.Sprintf("%d observations synthesized", len(record.Observations)),
		}
	}
	s.mu.Unlock()

	if event != nil {
		s.emit(ctx, execCtx, *event)
	}
	return nil
}

// SynthesisHistory returns all completed synthesis records.
func (s *SynthesisTracker) SynthesisHistory() []SynthesisRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]SynthesisRecord(nil), s.synthesisHistory...)
}

func (s *SynthesisTracker) emit(ctx context.Context, execCtx *ExecutionContext, event SynthesisEvent) {
	if execCtx == nil {
		return
	}
	ch, ok := ctx.Value(planningEventChannelKey{}).(chan<- AgentEvent)
	if !ok || ch == nil {
		return
	}
	emit(ch, event)
}

// SynthesisPolicy blocks final answers while analysis remains incomplete.
type SynthesisPolicy struct {
	source synthesisStateSource
}

// NewSynthesisPolicy creates a reusable synthesis policy.
func NewSynthesisPolicy(source synthesisStateSource) *SynthesisPolicy {
	return &SynthesisPolicy{source: source}
}

// BeforeFinalAnswer implements FinalAnswerCallback.
func (p *SynthesisPolicy) BeforeFinalAnswer(
	ctx context.Context,
	execCtx *ExecutionContext,
	_ string,
) error {
	if p.source == nil || !p.source.HasIncompleteAnalysis() {
		return nil
	}
	if err := p.source.MarkSynthesisComplete(ctx, execCtx); err != nil {
		return err
	}
	return fmt.Errorf("Do not answer yet. Incomplete analysis remains. Synthesize the observations before providing a final answer.")
}
