package agent

import "errors"

// ErrMaxStepsReached is returned when Run exhausts maxSteps without a final answer.
var ErrMaxStepsReached = errors.New("agent: max steps reached without final answer")

// Result is the output of a successful Agent.Run() call.
type Result struct {
	// Output is the final answer produced by the LLM.
	Output string
	// ToolCalled reports whether at least one tool was invoked during the run.
	ToolCalled bool
	// Context is the full execution history for this run.
	Context *ExecutionContext
}
