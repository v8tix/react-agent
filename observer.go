package agent

import (
	"context"
	"time"

	"github.com/v8tix/react-agent/model"
)

// Observer receives lifecycle hooks for every agent run.
// All methods are called synchronously in the agent's goroutine.
// Implementations must not block; use a buffered channel or goroutine
// to hand off to a background writer.
//
// # Decorator pattern
//
// Embed NoopObserver in your struct and override only the hooks you need:
//
//	type LogObserver struct {
//	    agent.NoopObserver
//	    logger *slog.Logger
//	}
//
//	func (o *LogObserver) OnLLMCall(_ context.Context, runID string, _ model.Request, _ model.Response, latency time.Duration, err error) {
//	    o.logger.Info("llm call", "run_id", runID, "latency_ms", latency.Milliseconds(), "err", err)
//	}
type Observer interface {
	// OnRunStart is called once before the ReAct loop begins.
	OnRunStart(ctx context.Context, runID string, userMessage string)

	// OnRunEnd is called once when Run returns, whether or not it succeeded.
	// result is nil when err is non-nil.
	OnRunEnd(ctx context.Context, runID string, result *Result, err error)

	// OnStepStart is called at the beginning of each Think→Act cycle.
	OnStepStart(ctx context.Context, runID string, step int)

	// OnStepEnd is called after each Think→Act cycle, with the step error if any.
	OnStepEnd(ctx context.Context, runID string, step int, err error)

	// OnLLMCall is called after every Generate() call, including on error.
	// latency covers only the LLM network round-trip.
	OnLLMCall(ctx context.Context, runID string, req model.Request, resp model.Response, latency time.Duration, err error)

	// OnToolExecution is called after every executor.Execute() call, including on error.
	// results is nil when err is non-nil. latency is 0 when executor is nil.
	OnToolExecution(ctx context.Context, runID string, calls []model.ToolCall, results []model.ToolResult, latency time.Duration, err error)
}

// NoopObserver is an Observer that discards all events.
// Embed it in your own observer to implement only the hooks you care about.
type NoopObserver struct{}

func (NoopObserver) OnRunStart(_ context.Context, _ string, _ string) {}
func (NoopObserver) OnRunEnd(_ context.Context, _ string, _ *Result, _ error) {}
func (NoopObserver) OnStepStart(_ context.Context, _ string, _ int)   {}
func (NoopObserver) OnStepEnd(_ context.Context, _ string, _ int, _ error) {}
func (NoopObserver) OnLLMCall(_ context.Context, _ string, _ model.Request, _ model.Response, _ time.Duration, _ error) {
}
func (NoopObserver) OnToolExecution(_ context.Context, _ string, _ []model.ToolCall, _ []model.ToolResult, _ time.Duration, _ error) {
}
