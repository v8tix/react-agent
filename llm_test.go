// Whitebox tests for llm.go — must be package agent to access unexported functions.
package agent

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/v8tix/react-agent/model"
)

// ─── stripThinkTokens ────────────────────────────────────────────────────────

func TestStripThinkTokens(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"single block", "<think>internal reasoning</think>final answer", "final answer"},
		{"no block unchanged", "plain response with no think block", "plain response with no think block"},
		{"multiple blocks all removed", "<think>step 1</think>middle<think>step 2</think>end", "middleend"},
		{"multiline block", "<think>\nline 1\nline 2\n</think>answer", "answer"},
		{"empty input", "", ""},
		{"only block returns empty", "<think>everything is internal</think>", ""},
		{"unclosed tag no change", "<think>never closed — model truncated", "<think>never closed — model truncated"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stripThinkTokens(tt.input); got != tt.want {
				t.Errorf("want %q, got %q", tt.want, got)
			}
		})
	}
}

// ─── buildMessages ───────────────────────────────────────────────────────────

// wantToolCallParam describes one expected ToolCall entry inside an
// AssistantMessageParam. Used only when wantParam.calls is non-empty.
type wantToolCallParam struct {
	id   string
	name string
	args string
}

// wantParam describes the expected shape of a single message at a given
// position in the slice returned by buildMessages.
// Exactly one of system/user/assistant/tool should be true per entry.
type wantParam struct {
	system    bool
	user      bool
	assistant bool
	calls     []wantToolCallParam // assistant with tool calls; nil → plain assistant
	tool      bool
	toolID    string
}

func TestBuildMessages(t *testing.T) {
	tests := []struct {
		name         string
		instructions string
		events       []model.Event
		want         []wantParam
	}{
		{
			name:         "empty events — only system message",
			instructions: "You are helpful",
			want:         []wantParam{{system: true}},
		},
		{
			name: "empty instructions — no system message",
			events: []model.Event{
				{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "hi"}}},
			},
			want: []wantParam{{user: true}},
		},
		{
			name: "user event produces user message",
			events: []model.Event{
				{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "What is Go?"}}},
			},
			want: []wantParam{{user: true}},
		},
		{
			name: "agent plain text produces assistant message with no tool calls",
			events: []model.Event{
				{Author: "agent", Content: []model.ContentItem{model.Message{Role: "assistant", Content: "The answer is 42"}}},
			},
			want: []wantParam{{assistant: true}},
		},
		{
			// CRITICAL: OpenAI rejects requests where tool calls are not inside
			// AssistantMessageParam.ToolCalls. This validates the exact structural requirement.
			name: "agent tool calls produce AssistantMessageParam with ToolCalls",
			events: []model.Event{
				{Author: "agent", Content: []model.ContentItem{
					model.ToolCall{ID: "tc1", Name: "search", Arguments: []byte(`{"q":"Paris"}`)},
					model.ToolCall{ID: "tc2", Name: "weather", Arguments: []byte(`{"city":"Paris"}`)},
				}},
			},
			want: []wantParam{{
				assistant: true,
				calls: []wantToolCallParam{
					{"tc1", "search", `{"q":"Paris"}`},
					{"tc2", "weather", `{"city":"Paris"}`},
				},
			}},
		},
		{
			name: "tools event produces one tool message per result with correct IDs",
			events: []model.Event{
				{Author: "tools", Content: []model.ContentItem{
					model.ToolResult{ID: "tc1", Name: "search", Status: "success", Content: []string{"Paris"}},
					model.ToolResult{ID: "tc2", Name: "weather", Status: "success", Content: []string{"22°C"}},
				}},
			},
			want: []wantParam{
				{tool: true, toolID: "tc1"},
				{tool: true, toolID: "tc2"},
			},
		},
		{
			// Text is chain-of-thought; API expects AssistantMessageParam.ToolCalls.
			name: "agent mixed content — tool calls take precedence over plain text",
			events: []model.Event{
				{Author: "agent", Content: []model.ContentItem{
					model.Message{Role: "assistant", Content: "Let me look that up"},
					model.ToolCall{ID: "tc1", Name: "search", Arguments: []byte(`{}`)},
				}},
			},
			want: []wantParam{{
				assistant: true,
				calls:     []wantToolCallParam{{"tc1", "search", "{}"}},
			}},
		},
		{
			// Full think→act→observe. Ordering is critical — inversion causes API rejection.
			name:         "full ReAct conversation — correct message order",
			instructions: "You are helpful",
			events: []model.Event{
				{Author: "user", Content: []model.ContentItem{
					model.Message{Role: "user", Content: "Capital of France?"},
				}},
				{Author: "agent", Content: []model.ContentItem{
					model.ToolCall{ID: "tc1", Name: "search", Arguments: []byte(`{"q":"France capital"}`)},
				}},
				{Author: "tools", Content: []model.ContentItem{
					model.ToolResult{ID: "tc1", Name: "search", Status: "success", Content: []string{"Paris"}},
				}},
			},
			want: []wantParam{
				{system: true},
				{user: true},
				{assistant: true, calls: []wantToolCallParam{{"tc1", "search", `{"q":"France capital"}`}}},
				{tool: true, toolID: "tc1"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msgs := buildMessages(tt.instructions, tt.events)
			if len(msgs) != len(tt.want) {
				t.Fatalf("want %d messages, got %d", len(tt.want), len(msgs))
			}
			for i, w := range tt.want {
				msg := msgs[i]
				switch {
				case w.system:
					if msg.OfSystem == nil {
						t.Errorf("msg[%d]: want system, got nil OfSystem", i)
					}
				case w.user:
					if msg.OfUser == nil {
						t.Errorf("msg[%d]: want user, got nil OfUser", i)
					}
				case w.assistant:
					if msg.OfAssistant == nil {
						t.Errorf("msg[%d]: want assistant, got nil OfAssistant", i)
						continue
					}
					if len(msg.OfAssistant.ToolCalls) != len(w.calls) {
						t.Errorf("msg[%d]: want %d tool calls, got %d", i, len(w.calls), len(msg.OfAssistant.ToolCalls))
						continue
					}
					for j, wc := range w.calls {
						tc := msg.OfAssistant.ToolCalls[j]
						if tc.ID != wc.id || tc.Function.Name != wc.name {
							t.Errorf("msg[%d].calls[%d]: want {%s,%s}, got {%s,%s}", i, j, wc.id, wc.name, tc.ID, tc.Function.Name)
						}
						if tc.Function.Arguments != wc.args {
							t.Errorf("msg[%d].calls[%d] args: want %s, got %s", i, j, wc.args, tc.Function.Arguments)
						}
					}
				case w.tool:
					if msg.OfTool == nil {
						t.Errorf("msg[%d]: want tool, got nil OfTool", i)
						continue
					}
					if msg.OfTool.ToolCallID != w.toolID {
						t.Errorf("msg[%d]: want toolID=%s, got %s", i, w.toolID, msg.OfTool.ToolCallID)
					}
				}
			}
		})
	}
}

// ─── parseResponse ────────────────────────────────────────────────────────────

func TestParseResponse(t *testing.T) {
	type wantCall struct{ id, name, args string }

	tests := []struct {
		name        string
		input       openai.ChatCompletionMessage
		wantCalls   []wantCall // non-empty → expect ToolCall items in output
		wantRole    string     // non-empty → expect a single Message item
		wantContent string
	}{
		{
			name: "tool calls",
			input: openai.ChatCompletionMessage{
				ToolCalls: []openai.ChatCompletionMessageToolCall{
					{ID: "tc1", Function: openai.ChatCompletionMessageToolCallFunction{Name: "search", Arguments: `{"q":"Paris"}`}},
					{ID: "tc2", Function: openai.ChatCompletionMessageToolCallFunction{Name: "weather", Arguments: `{"city":"Paris"}`}},
				},
			},
			wantCalls: []wantCall{
				{"tc1", "search", `{"q":"Paris"}`},
				{"tc2", "weather", `{"city":"Paris"}`},
			},
		},
		{
			name:        "plain text",
			input:       openai.ChatCompletionMessage{Content: "The capital is Paris"},
			wantRole:    "assistant",
			wantContent: "The capital is Paris",
		},
		{
			name:        "think block stripped",
			input:       openai.ChatCompletionMessage{Content: "<think>Let me reason</think>The capital is Paris"},
			wantRole:    "assistant",
			wantContent: "The capital is Paris",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := parseResponse(tt.input)

			if len(tt.wantCalls) > 0 {
				if len(resp.Content) != len(tt.wantCalls) {
					t.Fatalf("want %d tool calls, got %d", len(tt.wantCalls), len(resp.Content))
				}
				for i, wc := range tt.wantCalls {
					tc, ok := resp.Content[i].(model.ToolCall)
					if !ok {
						t.Fatalf("content[%d]: want ToolCall", i)
					}
					if tc.ID != wc.id || tc.Name != wc.name {
						t.Errorf("content[%d]: want {%s,%s}, got {%s,%s}", i, wc.id, wc.name, tc.ID, tc.Name)
					}
					if string(tc.Arguments) != wc.args {
						t.Errorf("content[%d] args: want %s, got %s", i, wc.args, tc.Arguments)
					}
				}
				return
			}

			if len(resp.Content) != 1 {
				t.Fatalf("want 1 content item, got %d", len(resp.Content))
			}
			m, ok := resp.Content[0].(model.Message)
			if !ok {
				t.Fatal("content[0]: want Message")
			}
			if m.Role != tt.wantRole {
				t.Errorf("want role=%s, got %s", tt.wantRole, m.Role)
			}
			if m.Content != tt.wantContent {
				t.Errorf("want content=%q, got %q", tt.wantContent, m.Content)
			}
		})
	}
}
