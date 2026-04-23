package agent_test

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	agent "github.com/v8tix/react-agent"
	"github.com/v8tix/react-agent/model"
)

// ─── mockLLMClient ────────────────────────────────────────────────────────────

type mockLLMClient struct {
	responses []model.Response
	callCount int
}

func (m *mockLLMClient) Generate(_ context.Context, _ model.Request) (model.Response, error) {
	resp := m.responses[m.callCount%len(m.responses)]
	m.callCount++
	return resp, nil
}

// ─── ContentItem tests ────────────────────────────────────────────────────────

func TestMessage_Type(t *testing.T) {
	m := model.Message{Role: "user", Content: "hello"}
	if m.Type() != "message" {
		t.Fatalf("want message, got %s", m.Type())
	}
}

func TestToolCall_Type(t *testing.T) {
	tc := model.ToolCall{ID: "1", Name: "search", Arguments: json.RawMessage(`{}`)}
	if tc.Type() != "tool_call" {
		t.Fatalf("want tool_call, got %s", tc.Type())
	}
}

func TestToolResult_Type(t *testing.T) {
	tr := model.ToolResult{ID: "1", Name: "search", Status: "success"}
	if tr.Type() != "tool_result" {
		t.Fatalf("want tool_result, got %s", tr.Type())
	}
}

func TestMessage_JSONRoundTrip(t *testing.T) {
	original := model.Message{Role: "assistant", Content: "Paris"}
	b, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var decoded model.Message
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != original {
		t.Fatalf("want %+v, got %+v", original, decoded)
	}
}

// ─── ExecutionContext tests ───────────────────────────────────────────────────

func TestExecutionContext_AddEvent_AppendsInOrder(t *testing.T) {
	ec := agent.NewExecutionContextForTest()
	ec.AddEvent("user", model.Message{Role: "user", Content: "Q"})
	ec.AddEvent("agent", model.Message{Role: "assistant", Content: "A"})

	events := ec.Events()
	if len(events) != 2 {
		t.Fatalf("want 2 events, got %d", len(events))
	}
	if events[0].Author != "user" || events[1].Author != "agent" {
		t.Fatalf("unexpected authors: %s, %s", events[0].Author, events[1].Author)
	}
}

func TestExecutionContext_Events_DefensiveCopy(t *testing.T) {
	ec := agent.NewExecutionContextForTest()
	ec.AddEvent("user", model.Message{Role: "user", Content: "hi"})

	e1 := ec.Events()
	e1[0].Author = "mutated"
	e2 := ec.Events()

	if e2[0].Author != "user" {
		t.Fatal("mutation of returned slice affected internal state")
	}
}

func TestExecutionContext_IDNonEmpty(t *testing.T) {
	ec := agent.NewExecutionContextForTest()
	if ec.ID == "" {
		t.Fatal("expected non-empty ID")
	}
}

func TestExecutionContext_EventIDs_Unique(t *testing.T) {
	ec := agent.NewExecutionContextForTest()
	ec.AddEvent("user", model.Message{Role: "user", Content: "a"})
	ec.AddEvent("agent", model.Message{Role: "assistant", Content: "b"})

	events := ec.Events()
	if events[0].ID == events[1].ID {
		t.Fatal("event IDs should be unique")
	}
}

func TestExecutionContext_ConcurrentAddEvent(t *testing.T) {
	ec := agent.NewExecutionContextForTest()

	const n = 20
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			ec.AddEvent("agent", model.Message{Role: "assistant", Content: "concurrent"})
		}()
	}
	wg.Wait()

	if len(ec.Events()) != n {
		t.Fatalf("want %d events, got %d", n, len(ec.Events()))
	}
}

// ─── Agent tests ──────────────────────────────────────────────────────────────

func TestAgent_Run_NoTools_ReturnsAnswer(t *testing.T) {
	mock := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "Paris"}}},
		},
	}
	a := agent.New(mock, nil, nil, agent.WithInstructions("You are helpful"))
	result, err := a.Run(context.Background(), "Capital of France?")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Paris" {
		t.Fatalf("want Paris, got %s", result.Output)
	}
	if result.ToolCalled {
		t.Fatal("no tools should have been called")
	}
	if result.Context == nil {
		t.Fatal("expected non-nil context")
	}
}

func TestAgent_Run_MaxStepsExhausted_ReturnsError(t *testing.T) {
	mock := &mockLLMClient{
		responses: []model.Response{{Content: nil}},
	}
	a := agent.New(mock, nil, nil, agent.WithMaxSteps(2))
	_, err := a.Run(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
	// Verify the error wraps ErrMaxStepsReached
	unwrapped := err
	for unwrapped != nil {
		if unwrapped == agent.ErrMaxStepsReached {
			return
		}
		type unwrapper interface{ Unwrap() error }
		if u, ok := unwrapped.(unwrapper); ok {
			unwrapped = u.Unwrap()
		} else {
			break
		}
	}
	t.Fatalf("error does not wrap ErrMaxStepsReached: %v", err)
}

func TestAgent_Run_ContextContainsUserMessage(t *testing.T) {
	mock := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "ok"}}},
		},
	}
	a := agent.New(mock, nil, nil)
	result, err := a.Run(context.Background(), "test question")
	if err != nil {
		t.Fatal(err)
	}
	events := result.Context.Events()
	if len(events) == 0 {
		t.Fatal("expected events")
	}
	msg, ok := events[0].Content[0].(model.Message)
	if !ok {
		t.Fatal("first content item should be a Message")
	}
	if msg.Role != "user" || msg.Content != "test question" {
		t.Fatalf("unexpected message: %+v", msg)
	}
}

func TestAgent_Run_MultiStep_FinalAnswerOnStep2(t *testing.T) {
	mock := &mockLLMClient{
		responses: []model.Response{
			{Content: nil},
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "42"}}},
		},
	}
	a := agent.New(mock, nil, nil, agent.WithMaxSteps(5))
	result, err := a.Run(context.Background(), "What is 6*7?")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "42" {
		t.Fatalf("want 42, got %s", result.Output)
	}
	if result.Context.CurrentStep != 1 {
		t.Fatalf("want CurrentStep=1, got %d", result.Context.CurrentStep)
	}
}
