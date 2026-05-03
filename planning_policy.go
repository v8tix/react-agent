package agent

import (
	"context"
	"fmt"
)

type planRevisionSource interface {
	Revisions() []PlanRevision
}

// PlanningPolicy enforces a minimum number of planning revisions before the
// agent can return a final answer.
type PlanningPolicy struct {
	source       planRevisionSource
	minRevisions int
}

// NewPlanningPolicy creates a reusable planning-specific final-answer policy.
// A minimum revision count of zero or less defaults to two revisions.
func NewPlanningPolicy(source planRevisionSource, minRevisions int) *PlanningPolicy {
	if minRevisions <= 0 {
		minRevisions = 2
	}
	return &PlanningPolicy{
		source:       source,
		minRevisions: minRevisions,
	}
}

// BeforeFinalAnswer implements FinalAnswerCallback.
func (p *PlanningPolicy) BeforeFinalAnswer(
	_ context.Context,
	_ *ExecutionContext,
	_ string,
) error {
	if p.source != nil && len(p.source.Revisions()) >= p.minRevisions {
		return nil
	}
	return fmt.Errorf(
		"Do not answer yet. You must create a task list with %s and update it at least once before giving the final answer.",
		PlanningToolDefinition().Name,
	)
}
