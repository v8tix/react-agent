package agent

import (
	"context"

	"github.com/v8tix/react-agent/model"
)

// BeforeToolCallback can short-circuit a tool call before the executor runs.
// Returning a non-nil ToolResult skips executor execution for that call.
type BeforeToolCallback interface {
	BeforeTool(ctx context.Context, execCtx *ExecutionContext, call model.ToolCall) (*model.ToolResult, error)
}

// AfterToolCallback can replace a tool result after the executor (or a
// before-tool callback) produced it. Returning a non-nil ToolResult replaces the
// current result for that call.
type AfterToolCallback interface {
	AfterTool(ctx context.Context, execCtx *ExecutionContext, result model.ToolResult) (*model.ToolResult, error)
}
