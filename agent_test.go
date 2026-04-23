package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	agent "github.com/v8tix/react-agent"
	"github.com/v8tix/react-agent/model"
)

// ─── mocks ───────────────────────────────────────────────────────────────────

type mockLLMClient struct {
	responses []model.Response
	callCount int
	err       error
	mu        sync.Mutex
}

func (m *mockLLMClient) Generate(_ context.Context, _ model.Request) (model.Response, error) {
	if m.err != nil {
		return model.Response{}, m.err
	}
	m.mu.Lock()
	resp := m.responses[m.callCount%len(m.responses)]
	m.callCount++
	m.mu.Unlock()
	return resp, nil
}

type mockToolExecutor struct {
	results []model.ToolResult
	err     error
	calls   []model.ToolCall
	mu      sync.Mutex
}

func (m *mockToolExecutor) Execute(_ context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
	m.mu.Lock()
	m.calls = append(m.calls, calls...)
	m.mu.Unlock()
	if m.err != nil {
		return nil, m.err
	}
	if len(m.results) > 0 {
		return m.results, nil
	}
	out := make([]model.ToolResult, len(calls))
	for i, c := range calls {
		out[i] = model.ToolResult{ID: c.ID, Name: c.Name, Status: "success", Content: []string{"ok"}}
	}
	return out, nil
}

// ─── ContentItem ─────────────────────────────────────────────────────────────

func TestContentItem_Type(t *testing.T) {
	tests := []struct {
		name     string
		item     model.ContentItem
		wantType string
	}{
		{"message", model.Message{Role: "user", Content: "hello"}, "message"},
		{"tool_call", model.ToolCall{ID: "1", Name: "search", Arguments: json.RawMessage(`{}`)}, "tool_call"},
		{"tool_result", model.ToolResult{ID: "1", Name: "search", Status: "success"}, "tool_result"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.item.Type(); got != tt.wantType {
				t.Errorf("want %s, got %s", tt.wantType, got)
			}
		})
	}
}

func TestContentItem_JSONRoundTrip(t *testing.T) {
	t.Run("message", func(t *testing.T) {
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
			t.Errorf("want %+v, got %+v", original, decoded)
		}
	})

	t.Run("tool_call", func(t *testing.T) {
		original := model.ToolCall{
			ID:        "tc-123",
			Name:      "search",
			Arguments: json.RawMessage(`{"query":"Paris","limit":5}`),
		}
		b, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		var decoded model.ToolCall
		if err := json.Unmarshal(b, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.ID != original.ID || decoded.Name != original.Name {
			t.Errorf("want %+v, got %+v", original, decoded)
		}
		if string(decoded.Arguments) != string(original.Arguments) {
			t.Errorf("arguments mismatch: want %s, got %s", original.Arguments, decoded.Arguments)
		}
	})

	t.Run("tool_result", func(t *testing.T) {
		original := model.ToolResult{
			ID:      "tc-123",
			Name:    "search",
			Status:  "success",
			Content: []string{"result line 1", "result line 2"},
		}
		b, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		var decoded model.ToolResult
		if err := json.Unmarshal(b, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.ID != original.ID || decoded.Status != original.Status {
			t.Errorf("got %+v", decoded)
		}
		if len(decoded.Content) != 2 || decoded.Content[1] != "result line 2" {
			t.Errorf("content mismatch: %v", decoded.Content)
		}
	})
}

// ─── ExecutionContext ─────────────────────────────────────────────────────────

func TestExecutionContext(t *testing.T) {
	t.Run("IDNonEmpty", func(t *testing.T) {
		ec := agent.NewExecutionContextForTest()
		if ec.ID == "" {
			t.Fatal("expected non-empty ID")
		}
	})

	t.Run("AddEvent_AppendsInOrder", func(t *testing.T) {
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
	})

	t.Run("Events_DefensiveCopy", func(t *testing.T) {
		ec := agent.NewExecutionContextForTest()
		ec.AddEvent("user", model.Message{Role: "user", Content: "hi"})

		e1 := ec.Events()
		e1[0].Author = "mutated"
		e2 := ec.Events()

		if e2[0].Author != "user" {
			t.Fatal("mutation of returned slice affected internal state")
		}
	})

	t.Run("EventIDs_Unique", func(t *testing.T) {
		ec := agent.NewExecutionContextForTest()
		ec.AddEvent("user", model.Message{Role: "user", Content: "a"})
		ec.AddEvent("agent", model.Message{Role: "assistant", Content: "b"})

		events := ec.Events()
		if events[0].ID == events[1].ID {
			t.Fatal("event IDs should be unique")
		}
	})

	t.Run("ConcurrentAddEvent", func(t *testing.T) {
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
	})
}

func TestExecutionContext_IncrementStep(t *testing.T) {
	t.Run("sequential", func(t *testing.T) {
		ec := agent.NewExecutionContextForTest()
		if ec.CurrentStep != 0 {
			t.Fatalf("want initial step=0, got %d", ec.CurrentStep)
		}
		ec.IncrementStep()
		if ec.CurrentStep != 1 {
			t.Fatalf("want step=1 after first increment, got %d", ec.CurrentStep)
		}
		ec.IncrementStep()
		if ec.CurrentStep != 2 {
			t.Fatalf("want step=2 after second increment, got %d", ec.CurrentStep)
		}
	})

	t.Run("concurrent", func(t *testing.T) {
		ec := agent.NewExecutionContextForTest()
		const n = 50
		var wg sync.WaitGroup
		wg.Add(n)
		for i := 0; i < n; i++ {
			go func() {
				defer wg.Done()
				ec.IncrementStep()
			}()
		}
		wg.Wait()
		if ec.CurrentStep != n {
			t.Errorf("want CurrentStep=%d after %d concurrent increments, got %d", n, n, ec.CurrentStep)
		}
	})
}

// ─── Agent.Run — simple output cases ─────────────────────────────────────────

func TestAgent_Run(t *testing.T) {
	t.Run("NoTools_ReturnsAnswer", func(t *testing.T) {
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
	})

	t.Run("ContextContainsUserMessage", func(t *testing.T) {
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
	})

	t.Run("MultiStep_FinalAnswerOnStep2", func(t *testing.T) {
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
	})
}

// ─── Agent.Run — error propagation ───────────────────────────────────────────

func TestAgent_Run_ErrorPropagation(t *testing.T) {
	sentinelLLMErr := errors.New("provider unavailable")
	sentinelExecutorErr := errors.New("tool backend down")

	tests := []struct {
		name     string
		mock     *mockLLMClient
		executor *mockToolExecutor
		defs     []model.ToolDefinition
		wantErr  error
	}{
		{
			name:    "LLMError",
			mock:    &mockLLMClient{err: sentinelLLMErr},
			wantErr: sentinelLLMErr,
		},
		{
			name: "ExecutorError",
			mock: &mockLLMClient{responses: []model.Response{
				{Content: []model.ContentItem{
					model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{}`)},
				}},
			}},
			executor: &mockToolExecutor{err: sentinelExecutorErr},
			defs:     []model.ToolDefinition{{Name: "search"}},
			wantErr:  sentinelExecutorErr,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := agent.New(tt.mock, tt.defs, tt.executor)
			_, err := a.Run(context.Background(), "hello")
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("want errors.Is(err, %v), got: %v", tt.wantErr, err)
			}
		})
	}
}

// ─── Agent.Run — MaxSteps boundary ───────────────────────────────────────────

func TestAgent_Run_MaxSteps(t *testing.T) {
	tests := []struct {
		name          string
		maxSteps      int
		responses     []model.Response
		executor      *mockToolExecutor
		defs          []model.ToolDefinition
		wantNoLLMCall bool
	}{
		{
			// maxSteps=0: loop guard fires before first LLM call.
			name:     "zero_immediate_error",
			maxSteps: 0,
			responses: []model.Response{
				{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "answer"}}},
			},
			wantNoLLMCall: true,
		},
		{
			// maxSteps=2: LLM keeps returning nil content, loop exhausts.
			name:      "exhausted_with_nil_responses",
			maxSteps:  2,
			responses: []model.Response{{Content: nil}},
		},
		{
			// maxSteps=1: step 0 produces a tool call, IncrementStep → step=1 == maxSteps,
			// loop exits before the follow-up Think.
			name:     "tool_call_then_limit",
			maxSteps: 1,
			responses: []model.Response{
				{Content: []model.ContentItem{
					model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{}`)},
				}},
			},
			executor: &mockToolExecutor{},
			defs:     []model.ToolDefinition{{Name: "search"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockLLMClient{responses: tt.responses}
			a := agent.New(mock, tt.defs, tt.executor, agent.WithMaxSteps(tt.maxSteps))
			_, err := a.Run(context.Background(), "hi")

			if !errors.Is(err, agent.ErrMaxStepsReached) {
				t.Errorf("want ErrMaxStepsReached, got: %v", err)
			}
			if tt.wantNoLLMCall && mock.callCount != 0 {
				t.Errorf("want 0 LLM calls for maxSteps=0, got %d", mock.callCount)
			}
		})
	}
}

// ─── Full ReAct cycle — standalone (too complex for a table) ─────────────────

// TestAgent_Run_FullReActCycle is the core integration test:
// think → tool call → tool result → think → final answer.
// This is the only test that exercises Act() and validates ToolCalled=true.
func TestAgent_Run_FullReActCycle(t *testing.T) {
	toolCallStep := model.Response{Content: []model.ContentItem{
		model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)},
	}}
	finalStep := model.Response{Content: []model.ContentItem{
		model.Message{Role: "assistant", Content: "Paris is the capital of France"},
	}}
	mock := &mockLLMClient{responses: []model.Response{toolCallStep, finalStep}}
	executor := &mockToolExecutor{}
	defs := []model.ToolDefinition{{Name: "search", Description: "Search"}}

	a := agent.New(mock, defs, executor, agent.WithMaxSteps(10))
	result, err := a.Run(context.Background(), "Capital of France?")
	if err != nil {
		t.Fatal(err)
	}

	if result.Output != "Paris is the capital of France" {
		t.Errorf("want final answer, got %q", result.Output)
	}
	if !result.ToolCalled {
		t.Error("want ToolCalled=true")
	}

	executor.mu.Lock()
	calls := executor.calls
	executor.mu.Unlock()
	if len(calls) != 1 || calls[0].Name != "search" {
		t.Errorf("want 1 search call, got %+v", calls)
	}

	// Verify event structure: user → agent(tool call) → tools(result)
	events := result.Context.Events()
	if len(events) < 3 {
		t.Fatalf("want ≥3 events (user/agent/tools), got %d", len(events))
	}
	if events[0].Author != "user" {
		t.Errorf("events[0]: want user, got %s", events[0].Author)
	}
	if events[1].Author != "agent" {
		t.Errorf("events[1]: want agent, got %s", events[1].Author)
	}
	if _, ok := events[1].Content[0].(model.ToolCall); !ok {
		t.Error("events[1].Content[0]: want ToolCall")
	}
	if events[2].Author != "tools" {
		t.Errorf("events[2]: want tools, got %s", events[2].Author)
	}
	if _, ok := events[2].Content[0].(model.ToolResult); !ok {
		t.Error("events[2].Content[0]: want ToolResult")
	}
}

func TestAgent_Run_MultipleToolCallsInOneStep(t *testing.T) {
	toolCallStep := model.Response{Content: []model.ContentItem{
		model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"a"}`)},
		model.ToolCall{ID: "tc2", Name: "search", Arguments: json.RawMessage(`{"q":"b"}`)},
	}}
	finalStep := model.Response{Content: []model.ContentItem{
		model.Message{Role: "assistant", Content: "done"},
	}}
	mock := &mockLLMClient{responses: []model.Response{toolCallStep, finalStep}}
	executor := &mockToolExecutor{}
	defs := []model.ToolDefinition{{Name: "search"}}

	a := agent.New(mock, defs, executor)
	result, err := a.Run(context.Background(), "find a and b")
	if err != nil {
		t.Fatal(err)
	}

	executor.mu.Lock()
	calls := executor.calls
	executor.mu.Unlock()

	if len(calls) != 2 {
		t.Errorf("want 2 tool calls dispatched at once, got %d", len(calls))
	}
	if result.Output != "done" {
		t.Errorf("want done, got %s", result.Output)
	}
}

// ─── Concurrent safety ────────────────────────────────────────────────────────

func TestAgent_Run_Concurrent_NoSharedState(t *testing.T) {
	// N goroutines run concurrently on the same Agent; each must get its own
	// correct output without cross-contamination.
	const n = 20
	a := agent.New(
		&mockLLMClient{responses: []model.Response{
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "ok"}}},
		}},
		nil, nil,
	)

	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			result, err := a.Run(context.Background(), "ping")
			if err != nil {
				errs <- err
				return
			}
			if result.Output != "ok" {
				errs <- errors.New("unexpected output: " + result.Output)
				return
			}
			errs <- nil
		}()
	}

	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Errorf("concurrent run failed: %v", err)
		}
	}
}
