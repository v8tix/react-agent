package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestSlidingWindowStrategy_PreservesLastUserAndRecentEvents(t *testing.T) {
	req := model.Request{
		Events: []model.Event{
			{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "turn 1"}}},
			{Author: "agent", Content: []model.ContentItem{model.Message{Role: "assistant", Content: "a"}}},
			{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "turn 2"}}},
			{Author: "agent", Content: []model.ContentItem{model.Message{Role: "assistant", Content: "b"}}},
			{Author: "tools", Content: []model.ContentItem{model.ToolResult{ID: "tc-1", Name: "search", Status: "success", Content: []string{"c"}}}},
		},
	}

	if err := NewSlidingWindowStrategy(1).Optimize(context.Background(), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Events) != 2 {
		t.Fatalf("want 2 events, got %d", len(req.Events))
	}
	msg, ok := req.Events[0].Content[0].(model.Message)
	if !ok || msg.Content != "turn 2" {
		t.Fatalf("want last user message preserved, got %#v", req.Events[0].Content[0])
	}
}

func TestCompactionStrategy_CompressesConfiguredToolCallsAndResults(t *testing.T) {
	createArgs, _ := json.Marshal(map[string]any{"path": "report.md", "content": "full content"})
	readArgs, _ := json.Marshal(map[string]any{"path": "../secret.txt"})
	req := model.Request{
		Events: []model.Event{
			{Author: "agent", Content: []model.ContentItem{
				model.ToolCall{ID: "create-1", Name: "create_file", Arguments: createArgs},
				model.ToolCall{ID: "read-1", Name: "read_file", Arguments: readArgs},
			}},
			{Author: "tools", Content: []model.ContentItem{
				model.ToolResult{ID: "read-1", Name: "read_file", Status: "success", Content: []string{"secret"}},
			}},
		},
	}

	if err := NewCompactionStrategy().Optimize(context.Background(), &req); err != nil {
		t.Fatal(err)
	}
	result := req.Events[1].Content[0].(model.ToolResult)
	if result.Content[0] != "File content from [redacted-path]. Re-read if needed." {
		t.Fatalf("unexpected compacted result: %#v", result.Content)
	}
}

func TestMutatingLLMClient_AppliesMutatorsBeforeDelegating(t *testing.T) {
	calls := 0
	delegate := scriptedLLM(func(_ context.Context, _ model.Request) (model.Response, error) {
		calls++
		return model.Response{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "ok"}}}, nil
	})
	req := model.Request{
		Instructions: "system",
		Events:       []model.Event{{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "hello"}}}},
	}
	client := NewMutatingLLMClient(delegate, requestMutatorFunc(func(_ context.Context, req *model.Request) error {
		req.Instructions += "\nmutated"
		return nil
	}))

	if _, err := client.Generate(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("delegate callCount = %d, want 1", calls)
	}
}

func TestRequestTokenCounter_CountIncreasesWithMoreContent(t *testing.T) {
	counter, err := NewRequestTokenCounter("gpt-4o-mini")
	if err != nil {
		t.Fatal(err)
	}
	small := model.Request{Instructions: "hi"}
	large := model.Request{
		Instructions: "hi there with more content",
		Events:       []model.Event{{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "hello world"}}}},
		Tools:        []model.ToolDefinition{{Name: "search", Description: "search", Parameters: map[string]any{"type": "object"}}},
	}
	smallCount, err := counter.Count(small)
	if err != nil {
		t.Fatal(err)
	}
	largeCount, err := counter.Count(large)
	if err != nil {
		t.Fatal(err)
	}
	if largeCount <= smallCount {
		t.Fatalf("want large count > small count, got %d <= %d", largeCount, smallCount)
	}
}

type requestMutatorFunc func(context.Context, *model.Request) error

func (f requestMutatorFunc) Mutate(ctx context.Context, req *model.Request) error { return f(ctx, req) }
