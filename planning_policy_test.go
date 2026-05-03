package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestPlanningPolicy_RejectsWithoutEnoughRevisions(t *testing.T) {
	t.Parallel()

	executor := NewPlanningExecutor(nil)
	policy := NewPlanningPolicy(executor, 2)

	err := policy.BeforeFinalAnswer(context.Background(), NewExecutionContextForTest(), "too early")
	if err == nil {
		t.Fatal("expected rejection when the plan has not been revised")
	}
}

func TestPlanningPolicy_AcceptsAfterEnoughRevisions(t *testing.T) {
	t.Parallel()

	executor := NewPlanningExecutor(nil)
	first, err := MarshalPlanTasks([]PlanTask{
		{Content: "Find source", Status: PlanTaskPending},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}
	second, err := MarshalPlanTasks([]PlanTask{
		{Content: "Find source", Status: PlanTaskCompleted},
		{Content: "Draft answer", Status: PlanTaskInProgress},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	_, err = executor.Execute(context.Background(), []model.ToolCall{
		{ID: "tc-1", Name: PlanningToolDefinition().Name, Arguments: first},
		{ID: "tc-2", Name: PlanningToolDefinition().Name, Arguments: second},
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	policy := NewPlanningPolicy(executor, 2)
	if err := policy.BeforeFinalAnswer(context.Background(), NewExecutionContextForTest(), "allowed"); err != nil {
		t.Fatalf("BeforeFinalAnswer() error = %v", err)
	}
}

func TestPlanningPolicy_RevisionsExposeStructuredSnapshots(t *testing.T) {
	t.Parallel()

	executor := NewPlanningExecutor(nil)
	raw, err := MarshalPlanTasks([]PlanTask{
		{Content: "Find source", Status: PlanTaskInProgress},
		{Content: "Draft answer", Status: PlanTaskPending},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	_, err = executor.Execute(context.Background(), []model.ToolCall{{
		ID:        "tc-1",
		Name:      PlanningToolDefinition().Name,
		Arguments: raw,
	}})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	revisions := executor.Revisions()
	if len(revisions) != 1 {
		t.Fatalf("len(Revisions()) = %d, want 1", len(revisions))
	}
	if revisions[0].Index != 0 || revisions[0].TaskCount != 2 {
		t.Fatalf("Revisions()[0] = %#v, want index 0 and task count 2", revisions[0])
	}
	if revisions[0].Plan == "" {
		t.Fatal("expected captured plan text")
	}
}

func TestPlanningPolicy_UsesRevisionSourceInsteadOfConcreteExecutor(t *testing.T) {
	t.Parallel()

	policy := NewPlanningPolicy(staticRevisionSource{
		{Index: 0, Plan: "[ ] Find source", TaskCount: 1},
		{Index: 1, Plan: "[x] ~~Find source~~\n[>] **Draft answer**", TaskCount: 2},
	}, 2)

	if err := policy.BeforeFinalAnswer(context.Background(), NewExecutionContextForTest(), "allowed"); err != nil {
		t.Fatalf("BeforeFinalAnswer() error = %v", err)
	}
}

func TestPlanningPolicy_NilSourceRejectsWithoutPanicking(t *testing.T) {
	t.Parallel()

	policy := NewPlanningPolicy(nil, 2)
	err := policy.BeforeFinalAnswer(context.Background(), NewExecutionContextForTest(), "too early")
	if err == nil {
		t.Fatal("expected rejection for nil revision source")
	}
	if !strings.Contains(err.Error(), "create a task list") {
		t.Fatalf("unexpected error: %v", err)
	}
}

type staticRevisionSource []PlanRevision

func (s staticRevisionSource) Revisions() []PlanRevision {
	return []PlanRevision(s)
}
