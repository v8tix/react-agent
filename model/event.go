package model

import "time"

// Event is the atomic unit of history in an ExecutionContext.
// Author identifies the source: "user", "agent", or "tools".
// One event per step-participant — the LLM produces one agent event per Think,
// the tool runner produces one tools event per Act.
type Event struct {
	ID          string        `json:"id"`
	ExecutionID string        `json:"execution_id"`
	Timestamp   time.Time     `json:"timestamp"`
	Author      string        `json:"author"` // "user" | "agent" | "tools"
	Content     []ContentItem `json:"content"`
}
