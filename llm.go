package agent

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/v8tix/react-agent/model"
)

// thinkRe strips Qwen3 / reasoning-model chain-of-thought blocks so they don't
// accumulate in history and cause TPS collapse on long conversations.
var thinkRe = regexp.MustCompile(`(?s)<think>.*?</think>`)

func stripThinkTokens(s string) string {
	return strings.TrimSpace(thinkRe.ReplaceAllString(s, ""))
}

// ─── LLMClient interface ─────────────────────────────────────────────────────

// LLMClient abstracts communication with a language model.
// Implement this interface to support any LLM provider.
type LLMClient interface {
	Generate(ctx context.Context, req model.Request) (model.Response, error)
}

// ─── LiteLLMClient (openai-go adapter) ───────────────────────────────────────

// LiteLLMClient adapts the openai-go client to the LLMClient interface.
// Works with OpenAI directly or with a LiteLLM proxy (same API surface).
type LiteLLMClient struct {
	client *openai.Client
	model  openai.ChatModel
}

// NewLiteLLMClient creates a LiteLLMClient wrapping the provided openai-go client.
func NewLiteLLMClient(client *openai.Client, model openai.ChatModel) *LiteLLMClient {
	return &LiteLLMClient{client: client, model: model}
}

// Generate translates a Request into an OpenAI chat completion and maps the
// response back to ContentItem types.
func (c *LiteLLMClient) Generate(ctx context.Context, req model.Request) (model.Response, error) {
	messages := buildMessages(req.Instructions, req.Events)
	toolParams := toOpenAIToolParams(req.Tools)

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}

	resp, err := c.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:               c.model,
		Messages:            messages,
		Tools:               toolParams,
		MaxCompletionTokens: openai.Int(maxTokens),
	})
	if err != nil {
		return model.Response{}, fmt.Errorf("litellm generate: %w", err)
	}
	if len(resp.Choices) == 0 {
		return model.Response{}, fmt.Errorf("litellm generate: empty choices")
	}

	return parseResponse(resp.Choices[0].Message), nil
}

// ─── Message construction ─────────────────────────────────────────────────────

// buildMessages converts a system instruction + event log into the OpenAI
// message format. Events are processed by Author to preserve the required
// structure:
//
//   - "user" events  → UserMessage per Message item
//   - "agent" events → AssistantMessageParam with ToolCalls when the event
//     contains ToolCall items; plain AssistantMessage otherwise
//   - "tools" events → one ToolMessage per ToolResult item
//
// Processing by event (not by flat ContentItem) is critical: OpenAI rejects
// requests where tool calls appear outside an AssistantMessage.ToolCalls array.
func buildMessages(instructions string, events []model.Event) []openai.ChatCompletionMessageParamUnion {
	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(events)+1)

	if instructions != "" {
		msgs = append(msgs, openai.SystemMessage(instructions))
	}

	for _, event := range events {
		switch event.Author {
		case "user":
			for _, item := range event.Content {
				if m, ok := item.(model.Message); ok {
					msgs = append(msgs, openai.UserMessage(m.Content))
				}
			}

		case "agent":
			toolCalls := collectToolCalls(event.Content)
			if len(toolCalls) > 0 {
				openaiCalls := make([]openai.ChatCompletionMessageToolCallParam, len(toolCalls))
				for i, tc := range toolCalls {
					openaiCalls[i] = openai.ChatCompletionMessageToolCallParam{
						ID:   tc.ID,
						Type: "function",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: string(tc.Arguments),
						},
					}
				}
				msgs = append(msgs, openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: openaiCalls,
					},
				})
			} else {
				for _, item := range event.Content {
					if m, ok := item.(model.Message); ok && m.Role == "assistant" {
						msgs = append(msgs, openai.AssistantMessage(m.Content))
					}
				}
			}

		case "tools":
			for _, item := range event.Content {
				if tr, ok := item.(model.ToolResult); ok {
					msgs = append(msgs, openai.ToolMessage(
						strings.Join(tr.Content, "\n"),
						tr.ID,
					))
				}
			}
		}
	}

	return msgs
}

// parseResponse converts an OpenAI chat message into ContentItem types.
func parseResponse(msg openai.ChatCompletionMessage) model.Response {
	if len(msg.ToolCalls) > 0 {
		items := make([]model.ContentItem, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			items[i] = model.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: []byte(tc.Function.Arguments),
			}
		}
		return model.Response{Content: items}
	}

	text := stripThinkTokens(msg.Content)
	return model.Response{Content: []model.ContentItem{model.Message{Role: "assistant", Content: text}}}
}

// toOpenAIToolParams converts ToolDefinitions to the OpenAI tool param format.
func toOpenAIToolParams(defs []model.ToolDefinition) []openai.ChatCompletionToolParam {
	if len(defs) == 0 {
		return nil
	}
	params := make([]openai.ChatCompletionToolParam, len(defs))
	for i, def := range defs {
		params[i] = openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        def.Name,
				Description: param.Opt[string]{Value: def.Description},
				Parameters:  openai.FunctionParameters(def.Parameters),
				Strict:      param.Opt[bool]{Value: def.Strict},
			},
		}
	}
	return params
}

func collectToolCalls(items []model.ContentItem) []model.ToolCall {
	var out []model.ToolCall
	for _, item := range items {
		if tc, ok := item.(model.ToolCall); ok {
			out = append(out, tc)
		}
	}
	return out
}
