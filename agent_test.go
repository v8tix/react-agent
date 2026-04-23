package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/reactivex/rxgo/v2"
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
	a := agent.New(mock, nil, nil).WithInstructions("You are helpful")
	result, _, err := a.Run(context.Background(), "Capital of France?")
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
	result, _, err := a.Run(context.Background(), "test question")
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
	a := agent.New(mock, nil, nil).WithMaxSteps(5)
	result, _, err := a.Run(context.Background(), "What is 6*7?")
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
			_, _, err := a.Run(context.Background(), "hello")
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
			a := agent.New(mock, tt.defs, tt.executor).WithMaxSteps(tt.maxSteps)
			_, _, err := a.Run(context.Background(), "hi")

			if !errors.Is(err, agent.ErrMaxStepsReached) {
				t.Errorf("want ErrMaxStepsReached, got: %v", err)
			}
			if tt.wantNoLLMCall && mock.callCount != 0 {
				t.Errorf("want 0 LLM calls for maxSteps=0, got %d", mock.callCount)
			}
		})
	}
}

// WithMaxSteps must panic on invalid input (n < 1).
func TestWithMaxSteps_PanicsOnInvalidInput(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"zero", 0},
		{"negative", -1},
	}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "ok"}}},
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("WithMaxSteps(%d): expected panic, got none", tt.n)
				}
			}()
			agent.New(mock, nil, nil).WithMaxSteps(tt.n) // must panic
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

	a := agent.New(mock, defs, executor).WithMaxSteps(10)
	result, _, err := a.Run(context.Background(), "Capital of France?")
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
	result, _, err := a.Run(context.Background(), "find a and b")
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
			result, _, err := a.Run(context.Background(), "ping")
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
	_, _, err := a.Run(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error when executor is nil, got nil")
	}
}

// ─── Observable event stream ──────────────────────────────────────────────────

// countEvents drains the observable (via rxgo.Defer semantics: fresh replay each call)
// and returns a map of event-type label → count.
func countEvents(obs rxgo.Observable) map[string]int {
	counts := map[string]int{}
	for item := range obs.Observe() {
		if item.E != nil {
			continue
		}
		switch item.V.(type) {
		case agent.RunStartEvent:
			counts["run_start"]++
		case agent.RunEndEvent:
			counts["run_end"]++
		case agent.StepStartEvent:
			counts["step_start"]++
		case agent.StepEndEvent:
			counts["step_end"]++
		case agent.LLMCallEvent:
			counts["llm_call"]++
		case agent.ToolExecEvent:
			counts["tool_exec"]++
		}
	}
	return counts
}

// findRunEndEvent returns the RunEndEvent from the observable.
func findRunEndEvent(obs rxgo.Observable) (agent.RunEndEvent, bool) {
	for item := range obs.Observe() {
		if item.E != nil {
			continue
		}
		if e, ok := item.V.(agent.RunEndEvent); ok {
			return e, true
		}
	}
	return agent.RunEndEvent{}, false
}

func TestAgent_Run_EventStream(t *testing.T) {
	tests := []struct {
		name         string
		responses    []model.Response
		withExecutor bool
		defs         []model.ToolDefinition
		maxSteps     int
		wantErr      bool
		wantErrIs    error
		wantCounts   map[string]int
		wantRunID    bool // RunStartEvent.RunID must match result.Context.ID()
	}{
		{
			name: "no-tools single step",
			responses: []model.Response{
				{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "Paris"}}},
			},
			wantCounts: map[string]int{
				"run_start": 1, "run_end": 1,
				"step_start": 1, "step_end": 1,
				"llm_call": 1, "tool_exec": 0,
			},
			wantRunID: true,
		},
		{
			name: "full ReAct cycle — 2 steps 1 tool exec",
			responses: []model.Response{
				{Content: []model.ContentItem{
					model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{}`)},
				}},
				{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
			},
			withExecutor: true,
			defs:         []model.ToolDefinition{{Name: "search"}},
			wantCounts: map[string]int{
				"run_start": 1, "run_end": 1,
				"step_start": 2, "step_end": 2,
				"llm_call": 2, "tool_exec": 1,
			},
		},
		{
			name:       "LLM error — RunEndEvent carries error",
			wantErr:    true,
			wantCounts: map[string]int{"run_start": 1, "run_end": 1},
		},
		{
			name:       "max steps — RunEndEvent carries ErrMaxStepsReached",
			responses:  []model.Response{{Content: nil}},
			maxSteps:   2,
			wantErr:    true,
			wantErrIs:  agent.ErrMaxStepsReached,
			wantCounts: map[string]int{"run_start": 1, "run_end": 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var mock *mockLLMClient
			if tt.wantErr && len(tt.responses) == 0 && tt.wantErrIs == nil {
				// LLM error case
				mock = &mockLLMClient{err: errors.New("provider down")}
			} else {
				mock = &mockLLMClient{responses: tt.responses}
			}

			var executor model.ToolExecutor
			if tt.withExecutor {
				executor = &mockToolExecutor{}
			}

			a := agent.New(mock, tt.defs, executor)
			if tt.maxSteps > 0 {
				a = a.WithMaxSteps(tt.maxSteps)
			}

			result, events, err := a.Run(context.Background(), "test")

			if tt.wantErr {
				if err == nil {
					t.Fatal("want error, got nil")
				}
				if tt.wantErrIs != nil && !errors.Is(err, tt.wantErrIs) {
					t.Errorf("want %v, got %v", tt.wantErrIs, err)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}

			counts := countEvents(events)
			for key, want := range tt.wantCounts {
				if counts[key] != want {
					t.Errorf("event[%s]: want %d, got %d", key, want, counts[key])
				}
			}

			// Verify RunEndEvent error matches Run() error
			runEnd, ok := findRunEndEvent(events)
			if !ok {
				t.Error("RunEndEvent missing from observable")
			} else if tt.wantErr {
				if runEnd.Err == nil {
					t.Error("RunEndEvent.Err: want non-nil, got nil")
				}
				if tt.wantErrIs != nil && !errors.Is(runEnd.Err, tt.wantErrIs) {
					t.Errorf("RunEndEvent.Err: want %v, got %v", tt.wantErrIs, runEnd.Err)
				}
			} else {
				if runEnd.Err != nil {
					t.Errorf("RunEndEvent.Err: want nil, got %v", runEnd.Err)
				}
			}

			// Verify RunStartEvent.RunID matches result context
			if tt.wantRunID && result != nil {
				for item := range events.Observe() {
					if item.E != nil {
						continue
					}
					if e, ok := item.V.(agent.RunStartEvent); ok {
						if e.RunID != result.Context.ID() {
							t.Errorf("RunStartEvent.RunID=%s, result.Context.ID()=%s", e.RunID, result.Context.ID())
						}
					}
				}
			}
		})
	}
}

// TestAgent_Run_ObservableIsReplayable verifies that multiple Observe() calls
// on the same observable each receive all events (rxgo.Defer semantics).
func TestAgent_Run_ObservableIsReplayable(t *testing.T) {
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "Paris"}}},
	}}
	a := agent.New(mock, nil, nil)
	_, events, err := a.Run(context.Background(), "Capital?")
	if err != nil {
		t.Fatal(err)
	}

	c1 := countEvents(events)
	c2 := countEvents(events) // second subscription — must replay
	if c1["run_start"] != 1 || c2["run_start"] != 1 {
		t.Errorf("replay: first=%v, second=%v — want run_start=1 each", c1, c2)
	}
}
