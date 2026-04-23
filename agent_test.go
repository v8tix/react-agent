package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

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

// ─── ContentItem.Type ─────────────────────────────────────────────────────────

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

// ─── ContentItem JSON round-trip ──────────────────────────────────────────────
// Kept as standalone functions: each type has a different assertion shape
// (Message uses == comparison, ToolCall needs Arguments comparison, ToolResult
// needs slice element checks), so they cannot share a uniform assertion loop.

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
		t.Errorf("want %+v, got %+v", original, decoded)
	}
}

func TestToolCall_JSONRoundTrip(t *testing.T) {
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
}

func TestToolResult_JSONRoundTrip(t *testing.T) {
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
}

// ─── ExecutionContext ─────────────────────────────────────────────────────────
// Kept as standalone functions: each tests a distinct, non-overlapping property
// of ExecutionContext that cannot be expressed as input→output rows.

func TestExecutionContext_IDNonEmpty(t *testing.T) {
	ec := agent.NewExecutionContextForTest()
	if ec.ID() == "" {
		t.Fatal("expected non-empty ID")
	}
}

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

func TestExecutionContext_SetGetState(t *testing.T) {
	ec := agent.NewExecutionContextForTest()

	// key absent → not found
	_, ok := ec.GetState("missing")
	if ok {
		t.Fatal("want ok=false for missing key")
	}

	// set then get
	ec.SetState("model", "gpt-4o")
	v, ok := ec.GetState("model")
	if !ok {
		t.Fatal("want ok=true after SetState")
	}
	if v != "model" {
		// v should equal the value we stored
		_ = v
	}
	if s, _ := v.(string); s != "gpt-4o" {
		t.Errorf("want gpt-4o, got %v", v)
	}

	// overwrite
	ec.SetState("model", "claude-3")
	v2, _ := ec.GetState("model")
	if s, _ := v2.(string); s != "claude-3" {
		t.Errorf("want claude-3 after overwrite, got %v", v2)
	}
}

// ─── ExecutionContext.IncrementStep ───────────────────────────────────────────

func TestExecutionContext_IncrementStep_Sequential(t *testing.T) {
	ec := agent.NewExecutionContextForTest()
	if ec.CurrentStep() != 0 {
		t.Fatalf("want initial step=0, got %d", ec.CurrentStep())
	}
	ec.IncrementStep()
	if ec.CurrentStep() != 1 {
		t.Fatalf("want step=1 after first increment, got %d", ec.CurrentStep())
	}
	ec.IncrementStep()
	if ec.CurrentStep() != 2 {
		t.Fatalf("want step=2 after second increment, got %d", ec.CurrentStep())
	}
}

func TestExecutionContext_IncrementStep_Concurrent(t *testing.T) {
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
	if ec.CurrentStep() != n {
		t.Errorf("want CurrentStep=%d after %d concurrent increments, got %d", n, n, ec.CurrentStep())
	}
}

// ─── Agent.Run — output and step count ───────────────────────────────────────
// Kept as standalone functions: each verifies a different aspect of Run()
// (output value, event structure, step counter) that cannot share assertion code.

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
	if result.Context.CurrentStep() != 1 {
		t.Fatalf("want CurrentStep=1, got %d", result.Context.CurrentStep())
	}
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
			name:    "LLM error",
			mock:    &mockLLMClient{err: sentinelLLMErr},
			wantErr: sentinelLLMErr,
		},
		{
			name: "executor error",
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
			// maxSteps=2: LLM keeps returning nil content, loop exhausts.
			name:      "exhausted with nil responses",
			maxSteps:  2,
			responses: []model.Response{{Content: nil}},
		},
		{
			// maxSteps=1: step 0 produces a tool call, IncrementStep → step=1 == maxSteps,
			// loop exits before the follow-up Think.
			name:     "tool call then limit",
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

// WithMaxSteps must panic immediately (before New) when n < 1.
func TestWithMaxSteps_PanicsOnInvalidInput(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"zero", 0},
		{"negative", -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("WithMaxSteps(%d): expected panic, got none", tt.n)
				}
			}()
			agent.WithMaxSteps(tt.n) // must panic immediately
		})
	}
}

// ─── Full ReAct cycle — standalone ───────────────────────────────────────────

// TestAgent_Run_FullReActCycle is the core integration test:
// think → tool call → tool result → think → final answer.
// Kept standalone because it validates multiple inter-dependent properties
// (ToolCalled, event structure, executor invocation) not expressible as rows.
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

	events := result.Context.Events()
	if len(events) != 4 {
		t.Fatalf("want 4 events (user/agent-tools/tools/agent-answer), got %d", len(events))
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
	if events[3].Author != "agent" {
		t.Errorf("events[3]: want agent, got %s", events[3].Author)
	}
	if msg, ok := events[3].Content[0].(model.Message); !ok || msg.Role != "assistant" {
		t.Errorf("events[3].Content[0]: want assistant Message, got %T", events[3].Content[0])
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

// ─── Nil executor guard ───────────────────────────────────────────────────────

// TestAgent_Act_NilExecutor verifies that Act returns an error (not a panic)
// when executor is nil but the LLM returns tool calls.
func TestAgent_Act_NilExecutor(t *testing.T) {
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{
			model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{}`)},
		}},
	}}
	defs := []model.ToolDefinition{{Name: "search"}}

	// executor is nil — agent.New with nil executor and non-empty defs
	a := agent.New(mock, defs, nil)
	_, err := a.Run(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error when executor is nil, got nil")
	}
}

// ─── Observer ─────────────────────────────────────────────────────────────────

type testObserver struct {
	agent.NoopObserver // embed to get no-op implementations for hooks we don't override
	mu           sync.Mutex
	runStarts    int
	runEnds      int
	stepStarts   int
	stepEnds     int
	llmCalls     int
	toolExecs    int
	lastRunID    string
	lastRunErr   error
	lastStepErrs []error
}

func (o *testObserver) OnRunStart(_ context.Context, runID string, _ string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.runStarts++
	o.lastRunID = runID
}
func (o *testObserver) OnRunEnd(_ context.Context, _ string, _ *agent.Result, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.runEnds++
	o.lastRunErr = err
}
func (o *testObserver) OnStepStart(_ context.Context, _ string, _ int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.stepStarts++
}
func (o *testObserver) OnStepEnd(_ context.Context, _ string, _ int, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.stepEnds++
	o.lastStepErrs = append(o.lastStepErrs, err)
}
func (o *testObserver) OnLLMCall(_ context.Context, _ string, _ model.Request, _ model.Response, _ time.Duration, _ error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.llmCalls++
}
func (o *testObserver) OnToolExecution(_ context.Context, _ string, _ []model.ToolCall, _ []model.ToolResult, _ time.Duration, _ error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.toolExecs++
}

func TestAgent_Run_ObserverHooksFired(t *testing.T) {
	t.Run("no-tools single step", func(t *testing.T) {
		obs := &testObserver{}
		mock := &mockLLMClient{responses: []model.Response{
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "Paris"}}},
		}}
		a := agent.New(mock, nil, nil, agent.WithObserver(obs))
		result, err := a.Run(context.Background(), "Capital?")
		if err != nil {
			t.Fatal(err)
		}

		obs.mu.Lock()
		defer obs.mu.Unlock()
		if obs.runStarts != 1 {
			t.Errorf("OnRunStart: want 1, got %d", obs.runStarts)
		}
		if obs.runEnds != 1 {
			t.Errorf("OnRunEnd: want 1, got %d", obs.runEnds)
		}
		if obs.stepStarts != 1 {
			t.Errorf("OnStepStart: want 1, got %d", obs.stepStarts)
		}
		if obs.stepEnds != 1 {
			t.Errorf("OnStepEnd: want 1, got %d", obs.stepEnds)
		}
		if obs.llmCalls != 1 {
			t.Errorf("OnLLMCall: want 1, got %d", obs.llmCalls)
		}
		if obs.toolExecs != 0 {
			t.Errorf("OnToolExecution: want 0 (no tools), got %d", obs.toolExecs)
		}
		if obs.lastRunID == "" {
			t.Error("runID should be non-empty")
		}
		if obs.lastRunID != result.Context.ID() {
			t.Errorf("runID mismatch: observer got %s, context has %s", obs.lastRunID, result.Context.ID())
		}
		if obs.lastRunErr != nil {
			t.Errorf("OnRunEnd: want nil err, got %v", obs.lastRunErr)
		}
	})

	t.Run("full ReAct cycle — 2 steps, 1 tool exec", func(t *testing.T) {
		obs := &testObserver{}
		mock := &mockLLMClient{responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{}`)},
			}},
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
		}}
		executor := &mockToolExecutor{}
		defs := []model.ToolDefinition{{Name: "search"}}
		a := agent.New(mock, defs, executor, agent.WithObserver(obs))
		_, err := a.Run(context.Background(), "find it")
		if err != nil {
			t.Fatal(err)
		}

		obs.mu.Lock()
		defer obs.mu.Unlock()
		if obs.stepStarts != 2 {
			t.Errorf("OnStepStart: want 2, got %d", obs.stepStarts)
		}
		if obs.llmCalls != 2 {
			t.Errorf("OnLLMCall: want 2, got %d", obs.llmCalls)
		}
		if obs.toolExecs != 1 {
			t.Errorf("OnToolExecution: want 1, got %d", obs.toolExecs)
		}
	})

	t.Run("LLM error — OnRunEnd receives error", func(t *testing.T) {
		obs := &testObserver{}
		sentinelErr := errors.New("provider down")
		mock := &mockLLMClient{err: sentinelErr}
		a := agent.New(mock, nil, nil, agent.WithObserver(obs))
		_, err := a.Run(context.Background(), "hi")
		if err == nil {
			t.Fatal("expected error")
		}

		obs.mu.Lock()
		defer obs.mu.Unlock()
		if obs.runEnds != 1 {
			t.Errorf("OnRunEnd: want 1 even on error, got %d", obs.runEnds)
		}
		if obs.lastRunErr == nil {
			t.Error("OnRunEnd: want non-nil err on LLM failure")
		}
		if !errors.Is(obs.lastRunErr, sentinelErr) {
			t.Errorf("OnRunEnd: want sentinel error, got %v", obs.lastRunErr)
		}
	})

	t.Run("max steps — OnRunEnd receives ErrMaxStepsReached", func(t *testing.T) {
		obs := &testObserver{}
		mock := &mockLLMClient{responses: []model.Response{{Content: nil}}}
		a := agent.New(mock, nil, nil, agent.WithMaxSteps(2), agent.WithObserver(obs))
		_, err := a.Run(context.Background(), "hi")
		if err == nil {
			t.Fatal("expected ErrMaxStepsReached")
		}

		obs.mu.Lock()
		defer obs.mu.Unlock()
		if obs.runEnds != 1 {
			t.Errorf("OnRunEnd: want 1, got %d", obs.runEnds)
		}
		if !errors.Is(obs.lastRunErr, agent.ErrMaxStepsReached) {
			t.Errorf("OnRunEnd: want ErrMaxStepsReached, got %v", obs.lastRunErr)
		}
	})

	t.Run("nil observer is a no-op", func(t *testing.T) {
		mock := &mockLLMClient{responses: []model.Response{
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "ok"}}},
		}}
		// WithObserver(nil) must not panic and must keep the default NoopObserver
		a := agent.New(mock, nil, nil, agent.WithObserver(nil))
		result, err := a.Run(context.Background(), "ping")
		if err != nil {
			t.Fatal(err)
		}
		if result.Output != "ok" {
			t.Errorf("want ok, got %s", result.Output)
		}
	})
}
