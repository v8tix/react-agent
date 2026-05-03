package agent

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/v8tix/react-agent/model"
)

// ApprovalRule defines how a tool approval request should be shown to a human
// and what message should come back if the action is denied.
type ApprovalRule struct {
	MessageTemplate string
	DeniedMessage   string
}

// ApprovalPolicy decides whether a given tool requires human approval.
type ApprovalPolicy interface {
	RuleForTool(name string) (ApprovalRule, bool)
}

// StaticApprovalPolicy maps tool names directly to approval rules.
type StaticApprovalPolicy map[string]ApprovalRule

// RuleForTool returns the configured rule for name, if any.
func (p StaticApprovalPolicy) RuleForTool(name string) (ApprovalRule, bool) {
	rule, ok := p[name]
	return rule, ok
}

// ConfirmationCallback pauses execution before selected tools so an external
// UI, API, or human can approve or deny the action.
type ConfirmationCallback struct {
	policy ApprovalPolicy
	logger *slog.Logger
}

// NewConfirmationCallback creates a before-tool callback backed by an approval policy.
func NewConfirmationCallback(policy ApprovalPolicy) ConfirmationCallback {
	return ConfirmationCallback{policy: policy}
}

// WithLogger attaches structured approval lifecycle logs.
func (c ConfirmationCallback) WithLogger(logger *slog.Logger) ConfirmationCallback {
	c.logger = logger
	return c
}

// BeforeTool requests approval for matching tools and redacts sensitive
// arguments before they are exposed in the interaction payload.
func (c ConfirmationCallback) BeforeTool(_ context.Context, execCtx *ExecutionContext, call model.ToolCall) (*model.ToolResult, error) {
	if c.policy == nil {
		return nil, nil
	}
	rule, ok := c.policy.RuleForTool(call.Name)
	if !ok {
		logDebug(c.logger, "approval_skipped", "tool", call.Name, "reason", "no_rule")
		return nil, nil
	}
	requestID := "approve-" + call.ID
	if response, ok := execCtx.InteractionResponse(requestID); ok {
		if response.Approved != nil && *response.Approved {
			logInfo(c.logger, "approval_reused", "tool", call.Name, "request_id", requestID, "approved", true)
			return nil, nil
		}
		denied := rule.DeniedMessage
		if denied == "" {
			denied = "Tool execution was rejected by user."
		}
		logInfo(c.logger, "approval_reused", "tool", call.Name, "request_id", requestID, "approved", false)
		return &model.ToolResult{Status: "error", Content: []string{denied}}, nil
	}
	prompt := rule.MessageTemplate
	if prompt == "" {
		prompt = fmt.Sprintf("Approve tool %s?", call.Name)
	}
	logInfo(c.logger, "approval_requested", "tool", call.Name, "request_id", requestID)
	return nil, Suspend(InteractionRequest{
		ID:         requestID,
		Kind:       "approval",
		Prompt:     prompt,
		ToolCallID: call.ID,
		ToolName:   call.Name,
		Payload: map[string]any{
			"arguments": redactToolArguments(call.Arguments),
		},
	})
}
