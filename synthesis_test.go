package agent

import (
	"context"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestSynthesisTracker_RecordsObservationsAndSyntheses(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	execCtx := NewExecutionContextForTest()

	// Record a tool observation
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"Paris is the capital of France"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}

	if !tracker.HasIncompleteAnalysis() {
		t.Fatal("expected incomplete analysis after recording observation")
	}

	if got := len(tracker.Observations()); got != 1 {
		t.Fatalf("len(Observations()) = %d, want 1", got)
	}
}

func TestSynthesisTracker_MarksSynthesisComplete(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	execCtx := NewExecutionContextForTest()

	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"found data"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}

	if err := tracker.MarkSynthesisComplete(context.Background(), execCtx); err != nil {
		t.Fatalf("MarkSynthesisComplete() error = %v", err)
	}

	if tracker.HasIncompleteAnalysis() {
		t.Fatal("expected analysis to be marked complete")
	}

	if got := len(tracker.SynthesisHistory()); got != 1 {
		t.Fatalf("len(SynthesisHistory()) = %d, want 1", got)
	}
}

func TestSynthesisPolicy_RejectsWhenIncompleteAnalysis(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	policy := NewSynthesisPolicy(tracker)
	execCtx := NewExecutionContextForTest()

	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"partial data"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}

	err := policy.BeforeFinalAnswer(context.Background(), execCtx, "incomplete answer")
	if err == nil {
		t.Fatal("expected rejection when analysis is incomplete")
	}
}

func TestSynthesisPolicy_AcceptsAfterCompleteSynthesis(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	policy := NewSynthesisPolicy(tracker)
	execCtx := NewExecutionContextForTest()

	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"complete data"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}

	if err := tracker.MarkSynthesisComplete(context.Background(), execCtx); err != nil {
		t.Fatalf("MarkSynthesisComplete() error = %v", err)
	}

	if err := policy.BeforeFinalAnswer(context.Background(), execCtx, "complete answer"); err != nil {
		t.Fatalf("BeforeFinalAnswer() error = %v", err)
	}
}

// Edge case: nil source in SynthesisPolicy
func TestSynthesisPolicy_AcceptsWithNilSource(t *testing.T) {
	t.Parallel()

	policy := NewSynthesisPolicy(nil)
	execCtx := NewExecutionContextForTest()

	// Should not panic or error with nil source
	err := policy.BeforeFinalAnswer(context.Background(), execCtx, "answer")
	if err != nil {
		t.Fatalf("BeforeFinalAnswer() with nil source error = %v, want nil", err)
	}
}

// Edge case: tracker handles nil execCtx gracefully
func TestSynthesisTracker_HandlesNilExecutionContext(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()

	// MarkSynthesisComplete with nil execCtx should not panic
	if err := tracker.MarkSynthesisComplete(context.Background(), nil); err != nil {
		t.Fatalf("MarkSynthesisComplete(nil execCtx) error = %v, want nil", err)
	}

	// No synthesis should be recorded
	if len(tracker.SynthesisHistory()) != 0 {
		t.Fatalf("len(SynthesisHistory()) = %d, want 0", len(tracker.SynthesisHistory()))
	}
}

// Edge case: tool results with non-success status are ignored
func TestSynthesisTracker_IgnoresNonSuccessResults(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	execCtx := NewExecutionContextForTest()

	// Record error result
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "error",
		Content: []string{"failed to search"},
	}); err != nil {
		t.Fatalf("AfterTool() with error status error = %v", err)
	}

	// Should have no observations
	if len(tracker.Observations()) != 0 {
		t.Fatalf("len(Observations()) = %d, want 0", len(tracker.Observations()))
	}

	// Should still have incomplete analysis marked as false (no success recorded)
	if tracker.HasIncompleteAnalysis() {
		t.Fatal("expected no incomplete analysis after ignoring error result")
	}

	// Record pending result
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-2",
		Name:    "search",
		Status:  "pending",
		Content: []string{"still loading"},
	}); err != nil {
		t.Fatalf("AfterTool() with pending status error = %v", err)
	}

	// Still no observations
	if len(tracker.Observations()) != 0 {
		t.Fatalf("len(Observations()) after pending = %d, want 0", len(tracker.Observations()))
	}
}

// Edge case: empty observations state after synthesis
func TestSynthesisTracker_ClearsObservationsAfterSynthesis(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	execCtx := NewExecutionContextForTest()

	// Record a tool observation
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"data 1"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}

	// Verify observation recorded
	if len(tracker.Observations()) != 1 {
		t.Fatalf("len(Observations()) before mark = %d, want 1", len(tracker.Observations()))
	}

	// Mark synthesis complete
	if err := tracker.MarkSynthesisComplete(context.Background(), execCtx); err != nil {
		t.Fatalf("MarkSynthesisComplete() error = %v", err)
	}

	// Observations should be cleared from working set
	if len(tracker.Observations()) != 0 {
		t.Fatalf("len(Observations()) after mark = %d, want 0", len(tracker.Observations()))
	}

	// But history should contain the synthesis
	if len(tracker.SynthesisHistory()) != 1 {
		t.Fatalf("len(SynthesisHistory()) = %d, want 1", len(tracker.SynthesisHistory()))
	}

	// Verify the history contains the observation
	if len(tracker.SynthesisHistory()[0].Observations) != 1 {
		t.Fatalf("len(SynthesisHistory()[0].Observations) = %d, want 1",
			len(tracker.SynthesisHistory()[0].Observations))
	}
}

// Edge case: multiple synthesis cycles don't interfere
func TestSynthesisTracker_MultipleSynthesisCycles(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	execCtx := NewExecutionContextForTest()

	// First cycle
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"data 1"},
	}); err != nil {
		t.Fatalf("AfterTool() cycle 1 error = %v", err)
	}

	if err := tracker.MarkSynthesisComplete(context.Background(), execCtx); err != nil {
		t.Fatalf("MarkSynthesisComplete() cycle 1 error = %v", err)
	}

	// Second cycle
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-2",
		Name:    "search",
		Status:  "success",
		Content: []string{"data 2"},
	}); err != nil {
		t.Fatalf("AfterTool() cycle 2 error = %v", err)
	}

	if err := tracker.MarkSynthesisComplete(context.Background(), execCtx); err != nil {
		t.Fatalf("MarkSynthesisComplete() cycle 2 error = %v", err)
	}

	// Should have 2 synthesis records
	if len(tracker.SynthesisHistory()) != 2 {
		t.Fatalf("len(SynthesisHistory()) = %d, want 2", len(tracker.SynthesisHistory()))
	}

	// Each should have 1 observation
	if len(tracker.SynthesisHistory()[0].Observations) != 1 {
		t.Fatalf("cycle 1 observation count = %d, want 1", len(tracker.SynthesisHistory()[0].Observations))
	}
	if len(tracker.SynthesisHistory()[1].Observations) != 1 {
		t.Fatalf("cycle 2 observation count = %d, want 1", len(tracker.SynthesisHistory()[1].Observations))
	}

	// Content should be different
	if tracker.SynthesisHistory()[0].Observations[0].Content == tracker.SynthesisHistory()[1].Observations[0].Content {
		t.Fatal("synthesis cycles should have different content")
	}
}

// Edge case: repeated rejections with deterministic behavior
func TestSynthesisPolicy_DeterministicRejectionLoop(t *testing.T) {
	t.Parallel()

	tracker := NewSynthesisTracker()
	policy := NewSynthesisPolicy(tracker)
	execCtx := NewExecutionContextForTest()

	// Record observation without completing synthesis
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-1",
		Name:    "search",
		Status:  "success",
		Content: []string{"incomplete data"},
	}); err != nil {
		t.Fatalf("AfterTool() error = %v", err)
	}

	// First rejection
	err1 := policy.BeforeFinalAnswer(context.Background(), execCtx, "incomplete answer 1")
	if err1 == nil {
		t.Fatal("expected rejection on first BeforeFinalAnswer")
	}

	// State after first rejection: analysis should still be incomplete but moved to history
	if tracker.HasIncompleteAnalysis() {
		t.Fatal("expected analysis to be marked complete after rejection")
	}

	// Record more observations
	if _, err := tracker.AfterTool(context.Background(), execCtx, model.ToolResult{
		ID:      "tc-2",
		Name:    "search",
		Status:  "success",
		Content: []string{"additional data"},
	}); err != nil {
		t.Fatalf("AfterTool() after rejection error = %v", err)
	}

	// Second rejection (same pattern)
	err2 := policy.BeforeFinalAnswer(context.Background(), execCtx, "incomplete answer 2")
	if err2 == nil {
		t.Fatal("expected rejection on second BeforeFinalAnswer")
	}

	// Error messages should be consistent
	if err1.Error() != err2.Error() {
		t.Fatalf("rejection messages should be consistent: %q vs %q", err1.Error(), err2.Error())
	}

	// Should have 2 synthesis records now
	if len(tracker.SynthesisHistory()) != 2 {
		t.Fatalf("len(SynthesisHistory()) = %d, want 2", len(tracker.SynthesisHistory()))
	}
}
