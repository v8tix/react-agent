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
		{
			name:  "single block",
			input: "<think>internal reasoning</think>final answer",
			want:  "final answer",
		},
		{
			name:  "no block unchanged",
			input: "plain response with no think block",
			want:  "plain response with no think block",
		},
		{
			name:  "multiple blocks all removed",
			input: "<think>step 1</think>middle<think>step 2</think>end",
			want:  "middleend",
		},
		{
			name:  "multiline block",
			input: "<think>\nline 1\nline 2\n</think>answer",
			want:  "answer",
		},
		{
			name:  "empty input",
			input: "",
			want:  "",
		},
		{
			name:  "only block returns empty",
			input: "<think>everything is internal</think>",
			want:  "",
		},
		{
			name:  "unclosed tag no change",
			input: "<think>never closed — model truncated",
			want:  "<think>never closed — model truncated",
		},
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

func TestBuildMessages(t *testing.T) {
	t.Run("EmptyEvents_OnlySystem", func(t *testing.T) {
		msgs := buildMessages("You are helpful", nil)
		if len(msgs) != 1 {
			t.Fatalf("want 1 message (system), got %d", len(msgs))
		}
		if msgs[0].OfSystem == nil {
			t.Fatal("want system message, got nil OfSystem")
		}
	})

	t.Run("EmptyInstructions_NoSystemMessage", func(t *testing.T) {
		msgs := buildMessages("", []model.Event{
			{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "hi"}}},
		})
		if len(msgs) != 1 {
			t.Fatalf("want 1 message (user only), got %d", len(msgs))
		}
		if msgs[0].OfUser == nil {
			t.Fatal("want user message, not system")
		}
	})

	t.Run("UserEvent_ProducesUserMessage", func(t *testing.T) {
		msgs := buildMessages("", []model.Event{
			{Author: "user", Content: []model.ContentItem{
				model.Message{Role: "user", Content: "What is Go?"},
			}},
		})
		if len(msgs) != 1 {
			t.Fatalf("want 1 message, got %d", len(msgs))
		}
		if msgs[0].OfUser == nil {
			t.Fatal("want OfUser, got nil")
		}
	})

	t.Run("AgentEvent_PlainText_ProducesAssistantMessage", func(t *testing.T) {
		msgs := buildMessages("", []model.Event{
			{Author: "agent", Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "The answer is 42"},
			}},
		})
		if len(msgs) != 1 {
			t.Fatalf("want 1 message, got %d", len(msgs))
		}
		if msgs[0].OfAssistant == nil {
			t.Fatal("want OfAssistant")
		}
		if len(msgs[0].OfAssistant.ToolCalls) != 0 {
			t.Errorf("want 0 ToolCalls for plain text, got %d", len(msgs[0].OfAssistant.ToolCalls))
		}
	})

	// CRITICAL: OpenAI rejects requests where tool calls are not inside
	// AssistantMessageParam.ToolCalls. This validates the exact structural requirement.
	t.Run("AgentEvent_WithToolCalls_ProducesAssistantToolCalls", func(t *testing.T) {
		msgs := buildMessages("", []model.Event{
			{
				Author: "agent",
				Content: []model.ContentItem{
					model.ToolCall{ID: "tc1", Name: "search", Arguments: []byte(`{"q":"Paris"}`)},
					model.ToolCall{ID: "tc2", Name: "weather", Arguments: []byte(`{"city":"Paris"}`)},
				},
			},
		})

		if len(msgs) != 1 {
			t.Fatalf("want 1 assistant message, got %d", len(msgs))
		}
		if msgs[0].OfAssistant == nil {
			t.Fatal("want OfAssistant (tool calls), got nil — OpenAI will reject this request")
		}
		if len(msgs[0].OfAssistant.ToolCalls) != 2 {
			t.Fatalf("want 2 tool calls in OfAssistant, got %d", len(msgs[0].OfAssistant.ToolCalls))
		}
		tc1 := msgs[0].OfAssistant.ToolCalls[0]
		if tc1.ID != "tc1" || tc1.Function.Name != "search" {
			t.Errorf("tc1: want {tc1, search}, got {%s, %s}", tc1.ID, tc1.Function.Name)
		}
		if tc1.Function.Arguments != `{"q":"Paris"}` {
			t.Errorf("tc1 args: want {\"q\":\"Paris\"}, got %s", tc1.Function.Arguments)
		}
		tc2 := msgs[0].OfAssistant.ToolCalls[1]
		if tc2.ID != "tc2" || tc2.Function.Name != "weather" {
			t.Errorf("tc2: want {tc2, weather}, got {%s, %s}", tc2.ID, tc2.Function.Name)
		}
	})

	t.Run("ToolsEvent_ProducesToolMessagesWithCorrectIDs", func(t *testing.T) {
		msgs := buildMessages("", []model.Event{
			{
				Author: "tools",
				Content: []model.ContentItem{
					model.ToolResult{ID: "tc1", Name: "search", Status: "success", Content: []string{"Paris is the capital"}},
					model.ToolResult{ID: "tc2", Name: "weather", Status: "success", Content: []string{"22°C", "sunny"}},
				},
			},
		})
		if len(msgs) != 2 {
			t.Fatalf("want 2 tool messages, got %d", len(msgs))
		}
		if msgs[0].OfTool == nil {
			t.Fatal("want OfTool for first result")
		}
		if msgs[0].OfTool.ToolCallID != "tc1" {
			t.Errorf("want ToolCallID=tc1, got %s", msgs[0].OfTool.ToolCallID)
		}
		if msgs[1].OfTool == nil {
			t.Fatal("want OfTool for second result")
		}
		if msgs[1].OfTool.ToolCallID != "tc2" {
			t.Errorf("want ToolCallID=tc2, got %s", msgs[1].OfTool.ToolCallID)
		}
	})

	// Agent event with both text and tool calls — tool calls take precedence
	// (text is chain-of-thought; API expects a single AssistantMessageParam with ToolCalls).
	t.Run("AgentEvent_MixedContent_ToolCallsTakePrecedence", func(t *testing.T) {
		msgs := buildMessages("", []model.Event{
			{
				Author: "agent",
				Content: []model.ContentItem{
					model.Message{Role: "assistant", Content: "Let me look that up"},
					model.ToolCall{ID: "tc1", Name: "search", Arguments: []byte(`{}`)},
				},
			},
		})
		if len(msgs) != 1 {
			t.Fatalf("want 1 message, got %d", len(msgs))
		}
		if msgs[0].OfAssistant == nil || len(msgs[0].OfAssistant.ToolCalls) != 1 {
			t.Error("want AssistantMessage with ToolCalls when event has both text and tool calls")
		}
	})

	// Full think→act→observe history. Ordering is critical — any inversion causes API rejection.
	t.Run("FullReActConversation", func(t *testing.T) {
		events := []model.Event{
			{Author: "user", Content: []model.ContentItem{
				model.Message{Role: "user", Content: "Capital of France?"},
			}},
			{Author: "agent", Content: []model.ContentItem{
				model.ToolCall{ID: "tc1", Name: "search", Arguments: []byte(`{"q":"France capital"}`)},
			}},
			{Author: "tools", Content: []model.ContentItem{
				model.ToolResult{ID: "tc1", Name: "search", Status: "success", Content: []string{"Paris"}},
			}},
		}

		msgs := buildMessages("You are helpful", events)

		// system + user + assistant(tool calls) + tool result = 4
		if len(msgs) != 4 {
			t.Fatalf("want 4 messages, got %d", len(msgs))
		}
		if msgs[0].OfSystem == nil {
			t.Error("msg[0]: want system")
		}
		if msgs[1].OfUser == nil {
			t.Error("msg[1]: want user")
		}
		if msgs[2].OfAssistant == nil || len(msgs[2].OfAssistant.ToolCalls) != 1 {
			t.Error("msg[2]: want assistant with 1 tool call")
		}
		if msgs[3].OfTool == nil {
			t.Error("msg[3]: want tool message")
		}
		if msgs[3].OfTool.ToolCallID != "tc1" {
			t.Errorf("msg[3]: want ToolCallID=tc1, got %s", msgs[3].OfTool.ToolCallID)
		}
	})
}

// ─── parseResponse ────────────────────────────────────────────────────────────

func TestParseResponse(t *testing.T) {
	t.Run("WithToolCalls_ReturnsToolCallItems", func(t *testing.T) {
		msg := openai.ChatCompletionMessage{
			ToolCalls: []openai.ChatCompletionMessageToolCall{
				{
					ID: "tc1",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "search",
						Arguments: `{"q":"Paris"}`,
					},
				},
				{
					ID: "tc2",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "weather",
						Arguments: `{"city":"Paris"}`,
					},
				},
			},
		}

		resp := parseResponse(msg)

		if len(resp.Content) != 2 {
			t.Fatalf("want 2 content items, got %d", len(resp.Content))
		}
		tc1, ok := resp.Content[0].(model.ToolCall)
		if !ok {
			t.Fatal("content[0] should be ToolCall")
		}
		if tc1.ID != "tc1" || tc1.Name != "search" {
			t.Errorf("tc1: want {tc1,search}, got {%s,%s}", tc1.ID, tc1.Name)
		}
		if string(tc1.Arguments) != `{"q":"Paris"}` {
			t.Errorf("tc1 args: want {\"q\":\"Paris\"}, got %s", tc1.Arguments)
		}
		tc2, ok := resp.Content[1].(model.ToolCall)
		if !ok {
			t.Fatal("content[1] should be ToolCall")
		}
		if tc2.ID != "tc2" || tc2.Name != "weather" {
			t.Errorf("tc2: want {tc2,weather}, got {%s,%s}", tc2.ID, tc2.Name)
		}
	})

	t.Run("PlainText_ReturnsAssistantMessage", func(t *testing.T) {
		msg := openai.ChatCompletionMessage{Content: "The capital is Paris"}

		resp := parseResponse(msg)

		if len(resp.Content) != 1 {
			t.Fatalf("want 1 content item, got %d", len(resp.Content))
		}
		m, ok := resp.Content[0].(model.Message)
		if !ok {
			t.Fatal("content[0] should be Message")
		}
		if m.Role != "assistant" {
			t.Errorf("want role=assistant, got %s", m.Role)
		}
		if m.Content != "The capital is Paris" {
			t.Errorf("want 'The capital is Paris', got %s", m.Content)
		}
	})

	t.Run("PlainText_ThinkBlockStripped", func(t *testing.T) {
		msg := openai.ChatCompletionMessage{
			Content: "<think>Let me reason</think>The capital is Paris",
		}

		resp := parseResponse(msg)

		m, ok := resp.Content[0].(model.Message)
		if !ok {
			t.Fatal("want Message")
		}
		if m.Content != "The capital is Paris" {
			t.Errorf("want think block stripped, got %q", m.Content)
		}
	})
}
