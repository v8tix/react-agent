package agent

import (
	"context"
	"strings"
)

// MemoryWriteDecision captures whether a long-term memory record should be stored.
type MemoryWriteDecision struct {
	ShouldStore bool
	Reason      string
	Score       float64
}

// MemoryWritePolicy decides whether a task memory is worth persisting.
type MemoryWritePolicy interface {
	Decide(context.Context, TaskMemory) (MemoryWriteDecision, error)
}

// ThresholdMemoryWritePolicy stores only memories whose heuristic score meets
// the configured threshold.
type ThresholdMemoryWritePolicy struct {
	threshold float64
}

// NewThresholdMemoryWritePolicy creates a simple heuristic write policy.
func NewThresholdMemoryWritePolicy(threshold float64) ThresholdMemoryWritePolicy {
	if threshold <= 0 {
		threshold = 0.5
	}
	return ThresholdMemoryWritePolicy{threshold: threshold}
}

// Decide scores the supplied memory and reports whether it should be stored.
func (p ThresholdMemoryWritePolicy) Decide(_ context.Context, memory TaskMemory) (MemoryWriteDecision, error) {
	score := memoryImportanceScore(memory)
	return MemoryWriteDecision{
		ShouldStore: score >= p.threshold,
		Reason:      "threshold policy",
		Score:       score,
	}, nil
}

func memoryImportanceScore(memory TaskMemory) float64 {
	var score float64
	if memory.IsCorrect {
		score += 0.4
	}
	if len(strings.TrimSpace(memory.TaskSummary)) >= 20 {
		score += 0.2
	}
	if len(strings.TrimSpace(memory.Approach)) >= 20 {
		score += 0.2
	}
	if len(strings.TrimSpace(memory.FinalAnswer)) >= 15 {
		score += 0.1
	}
	if strings.TrimSpace(memory.ErrorAnalysis) != "" {
		score += 0.2
	}
	if score > 1 {
		return 1
	}
	return score
}
