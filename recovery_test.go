package agent

import (
	"context"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestRecoveryTracker_RecordsFailuresAndRecoveries(t *testing.T) {
	t.Parallel()

	tracker := NewRecoveryTracker()
	execCtx := NewExecutionContextForTest()

	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "convert_units",
		Status:  "error",
		Content: []string{"temporary conversion failure"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}
	if !tracker.HasUnresolvedFailures() {
		t.Fatal("expected unresolved failures after error result")
	}
	if got := len(tracker.Failures()); got != 1 {
		t.Fatalf("len(Failures()) = %d, want 1", got)
	}

	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-2",
		Name:    "convert_units",
		Status:  "success",
		Content: []string{"3.048"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}
	if tracker.HasUnresolvedFailures() {
		t.Fatal("expected failures to be resolved after successful retry")
	}
	if got := len(tracker.Attempts()); got != 1 {
		t.Fatalf("len(Attempts()) = %d, want 1", got)
	}
}

func TestRecoveryPolicy_RejectsWhenUnresolvedFailuresRemain(t *testing.T) {
	t.Parallel()

	tracker := NewRecoveryTracker()
	policy := NewRecoveryPolicy(tracker)
	_, _ = tracker.AfterTool(context.Background(), NewExecutionContextForTest(), model.ToolResult{
		ID:      "tc-1",
		Name:    "convert_units",
		Status:  "error",
		Content: []string{"temporary conversion failure"},
	})

	err := policy.BeforeFinalAnswer(context.Background(), NewExecutionContextForTest(), "done")
	if err == nil {
		t.Fatal("expected unresolved recovery failure to reject final answer")
	}
}

func TestRecoveryPolicy_AcceptsAfterRecordedRecovery(t *testing.T) {
	t.Parallel()

	tracker := NewRecoveryTracker()
	policy := NewRecoveryPolicy(tracker)
	execCtx := NewExecutionContextForTest()
	_, _ = tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "convert_units",
		Status:  "error",
		Content: []string{"temporary conversion failure"},
	})
	_, _ = tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-2",
		Name:    "convert_units",
		Status:  "success",
		Content: []string{"3.048"},
	})

	if err := policy.BeforeFinalAnswer(context.Background(), execCtx, "done"); err != nil {
		t.Fatalf("BeforeFinalAnswer() error = %v", err)
	}
}

func TestRecoveryTracker_RequiresReflectionBeforeRetry(t *testing.T) {
	t.Parallel()

	tracker := NewRecoveryTracker()
	execCtx := NewExecutionContextForTest()
	_, _ = tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "convert_units",
		Status:  "error",
		Content: []string{"temporary conversion failure"},
	})

	override, err := tracker.BeforeTool(context.Background(), execCtx, model.ToolCall{
		ID:   "tc-2",
		Name: "convert_units",
	})
	if err != nil {
		t.Fatalf("BeforeTool() error = %v", err)
	}
	if override == nil {
		t.Fatal("expected retry to be blocked until reflection")
	}
	if !tracker.RequiresReflection() {
		t.Fatal("expected reflection to still be required")
	}
}

func TestRecoveryPolicy_RecordsReflectionText(t *testing.T) {
	t.Parallel()

	tracker := NewRecoveryTracker()
	policy := NewRecoveryPolicy(tracker)
	execCtx := NewExecutionContextForTest()
	_, _ = tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "convert_units",
		Status:  "error",
		Content: []string{"temporary conversion failure"},
	})

	err := policy.BeforeFinalAnswer(context.Background(), execCtx, "I should retry with the same conversion after the temporary failure.")
	if err == nil {
		t.Fatal("expected reflection message to be rejected back into the loop")
	}
	if got := tracker.LatestReflection(); got == "" {
		t.Fatal("expected reflection text to be recorded")
	}
	if tracker.RequiresReflection() {
		t.Fatal("expected reflection requirement to be cleared after recording reflection")
	}
}
