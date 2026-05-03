package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/reactivex/rxgo/v2"
	agent "github.com/v8tix/react-agent"
	"github.com/v8tix/react-agent/model"
)

type finalAnswerCallbackFunc func(context.Context, *agent.ExecutionContext, string) error

func (f finalAnswerCallbackFunc) BeforeFinalAnswer(ctx context.Context, execCtx *agent.ExecutionContext, answer string) error {
	return f(ctx, execCtx, answer)
}

type beforeToolCallbackFunc func(context.Context, *agent.ExecutionContext, model.ToolCall) (*model.ToolResult, error)

func (f beforeToolCallbackFunc) BeforeTool(ctx context.Context, execCtx *agent.ExecutionContext, call model.ToolCall) (*model.ToolResult, error) {
	return f(ctx, execCtx, call)
}

type blockingAnswerLLM struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (l *blockingAnswerLLM) Generate(_ context.Context, _ model.Request) (model.Response, error) {
	l.once.Do(func() { close(l.started) })
	<-l.release
	return model.Response{
		Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "done"},
		},
	}, nil
}

type dynamicToolsLLM struct {
	calls int
}

func (l *dynamicToolsLLM) Generate(_ context.Context, req model.Request) (model.Response, error) {
	l.calls++
	names := make([]string, 0, len(req.Tools))
	for _, tool := range req.Tools {
		names = append(names, tool.Name)
	}

	switch l.calls {
	case 1:
		if len(names) != 1 || names[0] != "first_tool" {
			return model.Response{}, fmt.Errorf("turn 1 tools = %v", names)
		}
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-first", Name: "first_tool", Arguments: json.RawMessage(`{}`)},
		}}, nil
	case 2:
		if len(names) != 1 || names[0] != "second_tool" {
			return model.Response{}, fmt.Errorf("turn 2 tools = %v", names)
		}
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "done"},
		}}, nil
	default:
		return model.Response{}, fmt.Errorf("unexpected call count %d", l.calls)
	}
}

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

type reflectionLoopLLM struct{}

func (l *reflectionLoopLLM) Generate(_ context.Context, req model.Request) (model.Response, error) {
	switch {
	case !hasToolResultStatus(req.Events, "error"):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-1", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
		}}, nil
	case !hasToolResultStatus(req.Events, "blocked"):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-2", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
		}}, nil
	case !hasUserMessageAfterBlockedResult(req.Events, "Do not call any tool yet. First reply with a short reflection on what failed and how you will correct the retry."):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-stuck", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
		}}, nil
	case !hasUserMessage(req.Events, "Thanks. Now retry the corrected approach before answering."):
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "I should retry the same conversion carefully."},
		}}, nil
	case !hasToolResultStatus(req.Events, "success"):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-3", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
		}}, nil
	default:
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "10 feet equals 3.048 meters."},
		}}, nil
	}
}

type planningStagnationLLM struct{}

type deferredPromptLLM struct{}

func (l *deferredPromptLLM) Generate(_ context.Context, req model.Request) (model.Response, error) {
	const prompt = "Finish the blocked tool step before continuing."
	switch {
	case !hasToolResultStatus(req.Events, "blocked"):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-1", Name: "gather_fact", Arguments: json.RawMessage(`{"topic":"record_time"}`)},
		}}, nil
	case !hasUserMessageAfterBlockedResult(req.Events, prompt):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-2", Name: "gather_fact", Arguments: json.RawMessage(`{"topic":"record_time"}`)},
		}}, nil
	default:
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "Prompt ordering stayed valid."},
		}}, nil
	}
}

func (l *planningStagnationLLM) Generate(_ context.Context, req model.Request) (model.Response, error) {
	stagnationPrompt := "Do not revise the plan again yet. First reply with a short reflection on why the plan is stalling and what you will change."
	revisionPrompt := "Thanks. Now revise your task list with create_tasks based on that reflection before answering."
	revisionCount := countToolResults(req.Events, agent.PlanningToolDefinition().Name, "success")

	switch {
	case revisionCount == 0:
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{
				ID:   "tc-plan-stall-1",
				Name: agent.PlanningToolDefinition().Name,
				Arguments: marshalPlanTasksOrFail(nil, []agent.PlanTask{
					{Content: "Find Kipchoge's official marathon world record time", Status: agent.PlanTaskInProgress},
					{Content: "Calculate average pace in km/h", Status: agent.PlanTaskPending},
				}),
			},
		}}, nil
	case revisionCount == 1:
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{
				ID:   "tc-plan-stall-2",
				Name: agent.PlanningToolDefinition().Name,
				Arguments: marshalPlanTasksOrFail(nil, []agent.PlanTask{
					{Content: "Find Kipchoge's official marathon world record time", Status: agent.PlanTaskCompleted},
					{Content: "Calculate average pace in km/h", Status: agent.PlanTaskInProgress},
				}),
			},
		}}, nil
	case !hasUserMessageAfterNthToolResult(req.Events, agent.PlanningToolDefinition().Name, "success", 2, stagnationPrompt):
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{
				ID:   "tc-plan-stall-3",
				Name: agent.PlanningToolDefinition().Name,
				Arguments: marshalPlanTasksOrFail(nil, []agent.PlanTask{
					{Content: "Convert 2:01:09 into hours", Status: agent.PlanTaskInProgress},
					{Content: "Divide 42.195 km by 2.019 hours", Status: agent.PlanTaskPending},
				}),
			},
		}}, nil
	case !hasUserMessage(req.Events, revisionPrompt):
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "I kept revising tasks without showing what changed. I should make the calculation and verification steps explicit before answering."},
		}}, nil
	case revisionCount == 2:
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{
				ID:   "tc-plan-stall-4",
				Name: agent.PlanningToolDefinition().Name,
				Arguments: marshalPlanTasksOrFail(nil, []agent.PlanTask{
					{Content: "Confirm Kipchoge's official marathon world record time: 2:01:09", Status: agent.PlanTaskCompleted},
					{Content: "Convert 2:01:09 into total hours", Status: agent.PlanTaskCompleted},
					{Content: "Calculate 42.195 km divided by 2.019 hours", Status: agent.PlanTaskCompleted},
					{Content: "Verify the pace rounds to about 20.9 km/h", Status: agent.PlanTaskCompleted},
				}),
			},
		}}, nil
	default:
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "Eliud Kipchoge's official marathon world record time is 2:01:09, which is about 20.9 km/h."},
		}}, nil
	}
}

type verificationGateLLM struct{}

func (l *verificationGateLLM) Generate(_ context.Context, req model.Request) (model.Response, error) {
	gatherPrompt := "Do not answer yet. Evidence is incomplete. Gather at least 2 supporting results before answering."
	evidenceCount := countToolResults(req.Events, "gather_fact", "success")

	switch {
	case evidenceCount == 0:
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-verify-1", Name: "gather_fact", Arguments: json.RawMessage(`{"topic":"record_time"}`)},
		}}, nil
	case !hasUserMessage(req.Events, gatherPrompt):
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "Kipchoge's pace is about 21 km/h."},
		}}, nil
	case evidenceCount == 1:
		return model.Response{Content: []model.ContentItem{
			model.ToolCall{ID: "tc-verify-2", Name: "gather_fact", Arguments: json.RawMessage(`{"topic":"distance"}`)},
		}}, nil
	default:
		return model.Response{Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "With two supporting facts gathered, Kipchoge's pace is about 20.9 km/h."},
		}}, nil
	}
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

type sequentialToolExecutor struct {
	batches [][]model.ToolResult
	call    int
	mu      sync.Mutex
}

func (s *sequentialToolExecutor) Execute(_ context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.call >= len(s.batches) {
		out := make([]model.ToolResult, len(calls))
		for i, c := range calls {
			out[i] = model.ToolResult{ID: c.ID, Name: c.Name, Status: "success", Content: []string{"ok"}}
		}
		return out, nil
	}
	out := s.batches[s.call]
	s.call++
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

func TestFinalAnswerCallback_RejectedAnswerContinuesLoop(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Too early"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	callback := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, answer string) error {
		if answer == "Too early" {
			return fmt.Errorf("do not answer yet")
		}
		return nil
	})

	a := agent.New(llm, nil, nil).WithFinalAnswerCallbacks(callback).WithMaxSteps(3)
	result, _, err := a.Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Allowed answer" {
		t.Fatalf("Output = %q, want Allowed answer", result.Output)
	}
}

func TestFinalAnswerCallback_AcceptsAnswer(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Accepted answer"},
			}},
		},
	}

	callback := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, _ string) error {
		return nil
	})

	a := agent.New(llm, nil, nil).WithFinalAnswerCallbacks(callback).WithMaxSteps(2)
	result, _, err := a.Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Accepted answer" {
		t.Fatalf("Output = %q, want Accepted answer", result.Output)
	}
}

func TestFinalAnswerCallback_RejectionAddsCorrectiveMessage(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Too early"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	callback := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, answer string) error {
		if answer == "Too early" {
			return fmt.Errorf("do not answer yet")
		}
		return nil
	})

	a := agent.New(llm, nil, nil).WithFinalAnswerCallbacks(callback).WithMaxSteps(3)
	result, _, err := a.Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	events := result.Context.Events()
	found := false
	for _, event := range events {
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" && msg.Content == "do not answer yet" {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("expected corrective user message in execution context")
	}
}

func TestFinalAnswerCallback_RejectedAnswerStaysInHistoryBeforeCorrection(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Too early"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	callback := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, answer string) error {
		if answer == "Too early" {
			return fmt.Errorf("do not answer yet")
		}
		return nil
	})

	a := agent.New(llm, nil, nil).WithFinalAnswerCallbacks(callback).WithMaxSteps(3)
	result, _, err := a.Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}

	events := result.Context.Events()
	seenRejectedAssistant := false
	seenCorrectionAfterRejectedAssistant := false
	for _, event := range events {
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if !ok {
				continue
			}
			if msg.Role == "assistant" && msg.Content == "Too early" {
				seenRejectedAssistant = true
			}
			if seenRejectedAssistant && msg.Role == "user" && msg.Content == "do not answer yet" {
				seenCorrectionAfterRejectedAssistant = true
			}
		}
	}
	if !seenRejectedAssistant {
		t.Fatal("expected rejected assistant answer in execution context")
	}
	if !seenCorrectionAfterRejectedAssistant {
		t.Fatal("expected corrective user message after rejected assistant answer")
	}
}

func TestAgent_WithPlanningPolicy_RejectedAnswerContinuesUntilPlanUpdated(t *testing.T) {
	t.Parallel()

	executor := agent.NewPlanningExecutor(nil)
	policy := agent.NewPlanningPolicy(executor, 2)

	first, err := agent.MarshalPlanTasks([]agent.PlanTask{
		{Content: "Find source", Status: agent.PlanTaskPending},
		{Content: "Draft answer", Status: agent.PlanTaskPending},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}
	second, err := agent.MarshalPlanTasks([]agent.PlanTask{
		{Content: "Find source", Status: agent.PlanTaskCompleted},
		{Content: "Draft answer", Status: agent.PlanTaskInProgress},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Too early"},
			}},
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-1", Name: agent.PlanningToolDefinition().Name, Arguments: first},
			}},
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-2", Name: agent.PlanningToolDefinition().Name, Arguments: second},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	result, _, err := agent.New(
		llm,
		[]model.ToolDefinition{agent.PlanningToolDefinition()},
		executor,
	).WithFinalAnswerCallbacks(policy).WithMaxSteps(5).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Allowed answer" {
		t.Fatalf("Output = %q, want Allowed answer", result.Output)
	}
}

func TestFinalAnswerCallback_EmitsPolicyEvent(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Too early"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	callback := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, answer string) error {
		if answer == "Too early" {
			return fmt.Errorf("do not answer yet")
		}
		return nil
	})

	_, events, err := agent.New(llm, nil, nil).WithFinalAnswerCallbacks(callback).WithMaxSteps(3).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}

	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) != 2 {
		t.Fatalf("len(policyEvents) = %d, want 2", len(policyEvents))
	}
	if policyEvents[0].Decision != agent.PolicyDecisionReject {
		t.Fatalf("first policy decision = %q, want reject", policyEvents[0].Decision)
	}
	if policyEvents[0].Reason != "do not answer yet" {
		t.Fatalf("first policy reason = %q, want do not answer yet", policyEvents[0].Reason)
	}
	if policyEvents[1].Decision != agent.PolicyDecisionAccept {
		t.Fatalf("second policy decision = %q, want accept", policyEvents[1].Decision)
	}
}

func TestPlanningExecutor_EmitsPlanRevisionEvent(t *testing.T) {
	t.Parallel()

	executor := agent.NewPlanningExecutor(nil)
	policy := agent.NewPlanningPolicy(executor, 2)
	first, err := agent.MarshalPlanTasks([]agent.PlanTask{
		{Content: "Find source", Status: agent.PlanTaskPending},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}
	second, err := agent.MarshalPlanTasks([]agent.PlanTask{
		{Content: "Find source", Status: agent.PlanTaskCompleted},
		{Content: "Draft answer", Status: agent.PlanTaskInProgress},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-1", Name: agent.PlanningToolDefinition().Name, Arguments: first},
			}},
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-2", Name: agent.PlanningToolDefinition().Name, Arguments: second},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	_, events, err := agent.New(
		llm,
		[]model.ToolDefinition{agent.PlanningToolDefinition()},
		executor,
	).WithFinalAnswerCallbacks(policy).WithMaxSteps(4).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}

	revisionEvents := collectPlanRevisionEvents(t, events)
	if len(revisionEvents) != 2 {
		t.Fatalf("len(revisionEvents) = %d, want 2", len(revisionEvents))
	}
	if revisionEvents[0].Revision.Index != 0 || revisionEvents[1].Revision.Index != 1 {
		t.Fatalf("unexpected revision indices: %#v", revisionEvents)
	}
	if revisionEvents[1].Revision.TaskCount != 2 {
		t.Fatalf("unexpected task count in second revision: %#v", revisionEvents[1].Revision)
	}
}

func TestAfterToolCallback_EmitsRecoveryEvents(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-1", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
			}},
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-2", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Done"},
			}},
		},
	}
	executor := &sequentialToolExecutor{
		batches: [][]model.ToolResult{{
			{ID: "tc-1", Name: "convert_units", Status: "error", Content: []string{"temporary failure"}},
		}, {
			{ID: "tc-2", Name: "convert_units", Status: "success", Content: []string{"3.048"}},
		}},
	}
	tracker := agent.NewRecoveryTracker()
	policy := agent.NewRecoveryPolicy(tracker)

	_, events, err := agent.New(
		llm,
		[]model.ToolDefinition{{Name: "convert_units"}},
		executor,
	).WithAfterToolCallbacks(tracker).WithFinalAnswerCallbacks(policy).WithMaxSteps(4).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}

	recoveryEvents := collectRecoveryEvents(t, events)
	if len(recoveryEvents) != 2 {
		t.Fatalf("len(recoveryEvents) = %d, want 2", len(recoveryEvents))
	}
	if recoveryEvents[0].Kind != agent.RecoveryEventFailureObserved {
		t.Fatalf("first recovery kind = %q, want failure_observed", recoveryEvents[0].Kind)
	}
	if recoveryEvents[1].Kind != agent.RecoveryEventRecovered {
		t.Fatalf("second recovery kind = %q, want recovered", recoveryEvents[1].Kind)
	}
}

func TestRecoveryFlow_RejectsDirectRetryUntilReflection(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-1", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
			}},
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-2", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "The first conversion failed temporarily, so I should retry the same conversion carefully."},
			}},
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-3", Name: "convert_units", Arguments: json.RawMessage(`{"value":10}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "10 feet equals 3.048 meters."},
			}},
		},
	}
	executor := &sequentialToolExecutor{
		batches: [][]model.ToolResult{{
			{ID: "tc-1", Name: "convert_units", Status: "error", Content: []string{"temporary failure"}},
		}, {
			{ID: "tc-3", Name: "convert_units", Status: "success", Content: []string{"3.048"}},
		}},
	}
	tracker := agent.NewRecoveryTracker()
	policy := agent.NewRecoveryPolicy(tracker)

	result, events, err := agent.New(
		llm,
		[]model.ToolDefinition{{Name: "convert_units"}},
		executor,
	).WithBeforeToolCallbacks(tracker).WithAfterToolCallbacks(tracker).WithFinalAnswerCallbacks(policy).WithMaxSteps(6).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "10 feet equals 3.048 meters." {
		t.Fatalf("Output = %q, want recovered answer", result.Output)
	}

	recoveryEvents := collectRecoveryEvents(t, events)
	if len(recoveryEvents) < 3 {
		t.Fatalf("len(recoveryEvents) = %d, want >= 3", len(recoveryEvents))
	}
	if recoveryEvents[1].Kind != agent.RecoveryEventReflectionRecorded {
		t.Fatalf("second recovery kind = %q, want reflection_recorded", recoveryEvents[1].Kind)
	}
	if tracker.LatestReflection() == "" {
		t.Fatal("expected tracker to contain reflection text")
	}

	eventsInContext := result.Context.Events()
	blockedRetryIndex := -1
	for i, event := range eventsInContext {
		if event.Author != "agent" {
			continue
		}
		for _, item := range event.Content {
			call, ok := item.(model.ToolCall)
			if !ok || call.ID != "tc-2" {
				continue
			}
			blockedRetryIndex = i
			break
		}
		if blockedRetryIndex >= 0 {
			break
		}
	}
	if blockedRetryIndex < 0 {
		t.Fatal("expected blocked retry tool call to be recorded")
	}
	if blockedRetryIndex+1 >= len(eventsInContext) {
		t.Fatal("expected tool result after blocked retry tool call")
	}
	if eventsInContext[blockedRetryIndex+1].Author != "tools" {
		t.Fatalf("event after blocked retry = %q, want tools", eventsInContext[blockedRetryIndex+1].Author)
	}
}

func TestRecoveryFlow_AddsPostBlockCorrectionForRetryingModels(t *testing.T) {
	t.Parallel()

	executor := &sequentialToolExecutor{
		batches: [][]model.ToolResult{{
			{ID: "tc-1", Name: "convert_units", Status: "error", Content: []string{"temporary failure"}},
		}, {
			{ID: "tc-3", Name: "convert_units", Status: "success", Content: []string{"3.048"}},
		}},
	}
	tracker := agent.NewRecoveryTracker()
	policy := agent.NewRecoveryPolicy(tracker)

	result, _, err := agent.New(
		&reflectionLoopLLM{},
		[]model.ToolDefinition{{Name: "convert_units"}},
		executor,
	).WithBeforeToolCallbacks(tracker).WithAfterToolCallbacks(tracker).WithFinalAnswerCallbacks(policy).WithMaxSteps(8).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "10 feet equals 3.048 meters." {
		t.Fatalf("Output = %q, want recovered answer", result.Output)
	}
	if tracker.LatestReflection() == "" {
		t.Fatal("expected tracker to capture reflection text")
	}
	if !hasUserMessageAfterBlockedResult(result.Context.Events(), "Do not call any tool yet. First reply with a short reflection on what failed and how you will correct the retry.") {
		t.Fatal("expected a post-block user correction after the blocked tool result")
	}
}

func TestAfterToolCallback_EmitsSynthesisEvents(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-1", Name: "search", Arguments: json.RawMessage(`{"query":"Paris"}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Paris is the capital"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Paris is the capital of France based on the search result."},
			}},
		},
	}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{
			ID: "tc-1", Name: "search", Status: "success", Content: []string{"Paris is the capital of France"},
		}},
	}
	tracker := agent.NewSynthesisTracker()
	policy := agent.NewSynthesisPolicy(tracker)

	_, events, err := agent.New(
		llm,
		[]model.ToolDefinition{{Name: "search"}},
		executor,
	).WithAfterToolCallbacks(tracker).WithFinalAnswerCallbacks(policy).WithMaxSteps(4).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}

	synthesisEvents := collectSynthesisEvents(t, events)
	if len(synthesisEvents) != 2 {
		t.Fatalf("len(synthesisEvents) = %d, want 2", len(synthesisEvents))
	}
	if synthesisEvents[0].Kind != agent.SynthesisEventObservationRecorded {
		t.Fatalf("first synthesis kind = %q, want observation_recorded", synthesisEvents[0].Kind)
	}
	if synthesisEvents[1].Kind != agent.SynthesisEventSynthesisComplete {
		t.Fatalf("second synthesis kind = %q, want synthesis_complete", synthesisEvents[1].Kind)
	}
}

func TestSynthesisPolicy_CreatesRejectionLoop(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "tc-1", Name: "search", Arguments: json.RawMessage(`{"query":"Paris"}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Paris is the capital"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Paris is the capital of France based on the search result."},
			}},
		},
	}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{
			ID: "tc-1", Name: "search", Status: "success", Content: []string{"Paris is the capital of France"},
		}},
	}
	tracker := agent.NewSynthesisTracker()
	policy := agent.NewSynthesisPolicy(tracker)

	result, events, err := agent.New(
		llm,
		[]model.ToolDefinition{{Name: "search"}},
		executor,
	).WithAfterToolCallbacks(tracker).WithFinalAnswerCallbacks(policy).WithMaxSteps(4).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Paris is the capital of France based on the search result." {
		t.Fatalf("Output = %q, want final synthesized answer", result.Output)
	}

	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) != 2 {
		t.Fatalf("len(policyEvents) = %d, want 2", len(policyEvents))
	}
	if policyEvents[0].Decision != agent.PolicyDecisionReject {
		t.Fatalf("first policy decision = %q, want reject", policyEvents[0].Decision)
	}
	if policyEvents[1].Decision != agent.PolicyDecisionAccept {
		t.Fatalf("second policy decision = %q, want accept", policyEvents[1].Decision)
	}
}

func TestAgent_Run_NoCallbacksEmitsNoPolicyEvents(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Allowed answer"},
			}},
		},
	}

	_, events, err := agent.New(llm, nil, nil).WithMaxSteps(2).Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if got := len(collectPolicyEvents(t, events)); got != 0 {
		t.Fatalf("len(policyEvents) = %d, want 0", got)
	}
}

func TestFinalAnswerCallback_FirstRejectStopsLaterPolicies(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Too early"},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Recovered"},
			}},
		},
	}

	secondCalled := 0
	first := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, _ string) error {
		return fmt.Errorf("stop here")
	})
	second := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, _ string) error {
		secondCalled++
		return nil
	})

	_, events, err := agent.New(llm, nil, nil).
		WithFinalAnswerCallbacks(first, second).
		WithMaxSteps(2).
		Run(context.Background(), "test")
	if !errors.Is(err, agent.ErrMaxStepsReached) {
		t.Fatalf("err = %v, want ErrMaxStepsReached", err)
	}
	if secondCalled != 0 {
		t.Fatalf("second policy calls = %d, want 0", secondCalled)
	}
	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) != 2 {
		t.Fatalf("len(policyEvents) = %d, want 2", len(policyEvents))
	}
	for i, event := range policyEvents {
		if event.Decision != agent.PolicyDecisionReject {
			t.Fatalf("policyEvents[%d].Decision = %q, want reject", i, event.Decision)
		}
		if event.PolicyName != "agent_test.finalAnswerCallbackFunc" {
			t.Fatalf("policyEvents[%d].PolicyName = %q, want first callback name", i, event.PolicyName)
		}
	}
}

func TestFinalAnswerCallback_MultipleAcceptingPoliciesEmitOneEventEach(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Accepted"},
			}},
		},
	}

	first := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, _ string) error { return nil })
	second := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, _ string) error { return nil })

	_, events, err := agent.New(llm, nil, nil).
		WithFinalAnswerCallbacks(first, second).
		WithMaxSteps(2).
		Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) != 2 {
		t.Fatalf("len(policyEvents) = %d, want 2", len(policyEvents))
	}
	for i, event := range policyEvents {
		if event.Decision != agent.PolicyDecisionAccept {
			t.Fatalf("policyEvents[%d].Decision = %q, want accept", i, event.Decision)
		}
	}
}

func TestFinalAnswerCallback_RejectionCanExhaustMaxSteps(t *testing.T) {
	t.Parallel()

	llm := &mockLLMClient{
		responses: []model.Response{
			{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "Too early"}}},
		},
	}
	rejecting := finalAnswerCallbackFunc(func(_ context.Context, _ *agent.ExecutionContext, _ string) error {
		return fmt.Errorf("not yet")
	})

	_, events, err := agent.New(llm, nil, nil).
		WithFinalAnswerCallbacks(rejecting).
		WithMaxSteps(2).
		Run(context.Background(), "test")
	if !errors.Is(err, agent.ErrMaxStepsReached) {
		t.Fatalf("err = %v, want ErrMaxStepsReached", err)
	}
	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) != 2 {
		t.Fatalf("len(policyEvents) = %d, want 2", len(policyEvents))
	}
	for i, event := range policyEvents {
		if event.Decision != agent.PolicyDecisionReject {
			t.Fatalf("policyEvents[%d].Decision = %q, want reject", i, event.Decision)
		}
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
		case agent.PolicyEvent:
			counts["policy"]++
		case agent.PlanRevisionEvent:
			counts["plan_revision"]++
		case agent.RecoveryEvent:
			counts["recovery"]++
		case agent.SynthesisEvent:
			counts["synthesis"]++
		}
	}
	return counts
}

func collectPolicyEvents(t *testing.T, events rxgo.Observable) []agent.PolicyEvent {
	t.Helper()
	var out []agent.PolicyEvent
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.PolicyEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

func hasToolResultStatus(events []model.Event, status string) bool {
	for _, event := range events {
		for _, item := range event.Content {
			result, ok := item.(model.ToolResult)
			if ok && result.Status == status {
				return true
			}
		}
	}
	return false
}

func hasAssistantMessage(events []model.Event, want string) bool {
	for _, event := range events {
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "assistant" && msg.Content == want {
				return true
			}
		}
	}
	return false
}

func hasUserMessage(events []model.Event, want string) bool {
	for _, event := range events {
		if event.Author != "user" {
			continue
		}
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" && msg.Content == want {
				return true
			}
		}
	}
	return false
}

func hasUserMessageAfterBlockedResult(events []model.Event, want string) bool {
	blockedSeen := false
	for _, event := range events {
		for _, item := range event.Content {
			result, ok := item.(model.ToolResult)
			if ok && result.Status == "blocked" {
				blockedSeen = true
			}
		}
		if !blockedSeen || event.Author != "user" {
			continue
		}
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" && msg.Content == want {
				return true
			}
		}
	}
	return false
}

func hasUserMessageAfterNthToolResult(events []model.Event, toolName, status string, n int, want string) bool {
	seen := 0
	for _, event := range events {
		for _, item := range event.Content {
			result, ok := item.(model.ToolResult)
			if ok && result.Name == toolName && result.Status == status {
				seen++
			}
		}
		if seen < n || event.Author != "user" {
			continue
		}
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" && msg.Content == want {
				return true
			}
		}
	}
	return false
}

func countToolResults(events []model.Event, toolName, status string) int {
	total := 0
	for _, event := range events {
		for _, item := range event.Content {
			result, ok := item.(model.ToolResult)
			if ok && result.Name == toolName && result.Status == status {
				total++
			}
		}
	}
	return total
}

func collectPlanRevisionEvents(t *testing.T, events rxgo.Observable) []agent.PlanRevisionEvent {
	t.Helper()
	var out []agent.PlanRevisionEvent
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.PlanRevisionEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

func collectRecoveryEvents(t *testing.T, events rxgo.Observable) []agent.RecoveryEvent {
	t.Helper()
	var out []agent.RecoveryEvent
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.RecoveryEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

func collectSynthesisEvents(t *testing.T, events rxgo.Observable) []agent.SynthesisEvent {
	t.Helper()
	var out []agent.SynthesisEvent
	if events == nil {
		return out
	}
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.SynthesisEvent); ok {
			out = append(out, event)
		}
	}
	return out
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

func TestAgent_Run_LiveEventSinkReceivesEventsBeforeRunCompletes(t *testing.T) {
	t.Parallel()

	started := make(chan struct{})
	release := make(chan struct{})
	liveEvents := make(chan agent.AgentEvent, 8)

	a := agent.New(&blockingAnswerLLM{
		started: started,
		release: release,
	}, nil, nil).WithLiveEventSink(func(event agent.AgentEvent) {
		liveEvents <- event
	})

	done := make(chan error, 1)
	go func() {
		_, _, err := a.Run(context.Background(), "hello")
		done <- err
	}()

	select {
	case <-started:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("LLM call did not start")
	}

	select {
	case err := <-done:
		t.Fatalf("Run() finished too early: %v", err)
	default:
	}

	select {
	case event := <-liveEvents:
		if _, ok := event.(agent.RunStartEvent); !ok {
			t.Fatalf("first live event = %T, want RunStartEvent", event)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("live event sink did not receive events before Run() completed")
	}

	close(release)

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Run() did not complete after releasing LLM")
	}
}

func TestAgent_Run_DynamicToolsCallbackFiltersToolsPerTurn(t *testing.T) {
	t.Parallel()

	llm := &dynamicToolsLLM{}
	executor := &mockToolExecutor{}
	turn := 0

	_, _, err := agent.New(
		llm,
		[]model.ToolDefinition{
			{Name: "first_tool"},
			{Name: "second_tool"},
		},
		executor,
	).WithDynamicToolsCallback(func(_ *agent.ExecutionContext) []model.ToolDefinition {
		turn++
		if turn == 1 {
			return []model.ToolDefinition{{Name: "first_tool"}}
		}
		return []model.ToolDefinition{{Name: "second_tool"}}
	}).Run(context.Background(), "use dynamic tools")
	if err != nil {
		t.Fatal(err)
	}

	if got := len(executor.calls); got != 1 {
		t.Fatalf("expected 1 executed tool call, got %d", got)
	}
	if executor.calls[0].Name != "first_tool" {
		t.Fatalf("expected first_tool execution, got %q", executor.calls[0].Name)
	}
}

// TestCollectSynthesisEvents_HandlesNilObservable verifies that collecting from nil observable is safe
func TestCollectSynthesisEvents_HandlesNilObservable(t *testing.T) {
	t.Parallel()

	// Should not panic on nil observable
	events := collectSynthesisEvents(t, nil)
	if len(events) != 0 {
		t.Fatalf("len(events) from nil observable = %d, want 0", len(events))
	}
}

// TestCollectSynthesisEvents_SkipsNonSynthesisEvents verifies proper event filtering
func TestCollectSynthesisEvents_SkipsNonSynthesisEvents(t *testing.T) {
	t.Parallel()

	// Just verify we can collect and filter events from a normal run
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "answer"}}},
	}}

	a := agent.New(mock, nil, nil)

	_, events, err := a.Run(context.Background(), "test query")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Collect all synthesis events (should be none in a simple run)
	synthesisEvents := collectSynthesisEvents(t, events)
	if len(synthesisEvents) != 0 {
		t.Fatalf("len(synthesisEvents) without tracker = %d, want 0", len(synthesisEvents))
	}

	// Now run with a synthesis tracker and verify events are collected
	tracker := agent.NewSynthesisTracker()
	executor := &mockToolExecutor{
		results: []model.ToolResult{
			{ID: "1", Name: "search", Status: "success", Content: []string{"data"}},
		},
	}

	mock2 := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{model.ToolCall{ID: "1", Name: "search", Arguments: []byte(`{}`)}}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "answer after data"}}},
	}}

	a2 := agent.New(mock2, []model.ToolDefinition{
		{Name: "search", Description: "search", Parameters: map[string]any{}},
	}, executor).WithAfterToolCallbacks(tracker)

	_, events2, err := a2.Run(context.Background(), "test")
	if err != nil {
		t.Fatalf("Run() with tracker error = %v", err)
	}

	synthesisEvents2 := collectSynthesisEvents(t, events2)
	// Should have observation_recorded event
	if len(synthesisEvents2) < 1 {
		t.Fatalf("len(synthesisEvents) with tracker = %d, want >= 1", len(synthesisEvents2))
	}

	// Verify event type
	if synthesisEvents2[0].Kind != agent.SynthesisEventObservationRecorded {
		t.Fatalf("first event kind = %q, want observation_recorded", synthesisEvents2[0].Kind)
	}
}

// TestAgent_EventCollectionAndReplay verifies events can be collected and replayed deterministically
func TestAgent_EventCollectionAndReplay(t *testing.T) {
	t.Parallel()

	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "answer 1"}}},
	}}

	a := agent.New(mock, nil, nil)
	result1, events1, err := a.Run(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}

	// Collect events
	eventList := []agent.AgentEvent{}
	for item := range events1.Observe() {
		if item.E != nil {
			continue
		}
		if e, ok := item.V.(agent.AgentEvent); ok {
			eventList = append(eventList, e)
		}
	}

	if len(eventList) == 0 {
		t.Fatal("expected events to be collected")
	}

	// Verify events can be observed again by re-calling Observe()
	eventList2 := []agent.AgentEvent{}
	for item := range events1.Observe() {
		if item.E != nil {
			continue
		}
		if e, ok := item.V.(agent.AgentEvent); ok {
			eventList2 = append(eventList2, e)
		}
	}

	if len(eventList) != len(eventList2) {
		t.Fatalf("replay mismatch: first=%d, second=%d", len(eventList), len(eventList2))
	}

	if result1.Output != "answer 1" {
		t.Fatalf("output = %q, want 'answer 1'", result1.Output)
	}
}

func TestPlanningReflection_UnifiedCycleRevisesAndRetries(t *testing.T) {
	t.Parallel()

	// Script the LLM to follow the unified planning/reflection cycle:
	// 1. Generate initial plan
	// 2. Attempt initial answer (will be rejected because plan not revised enough)
	// 3. After policy rejection, generate revised plan
	// 4. Generate final answer (accepted after second plan revision)
	llm := &mockLLMClient{
		responses: []model.Response{
			// Step 1: Initial plan
			{Content: []model.ContentItem{
				model.ToolCall{
					ID:   "tc-plan-1",
					Name: agent.PlanningToolDefinition().Name,
					Arguments: marshalPlanTasksOrFail(t, []agent.PlanTask{
						{Content: "Research the topic", Status: agent.PlanTaskInProgress},
						{Content: "Synthesize findings", Status: agent.PlanTaskPending},
					}),
				},
			}},
			// Step 2: Initial answer attempt (insufficient-progress signal in context)
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "The answer based on initial plan"},
			}},
			// Step 3: Revised plan (after reflection on insufficient progress)
			{Content: []model.ContentItem{
				model.ToolCall{
					ID:   "tc-plan-2",
					Name: agent.PlanningToolDefinition().Name,
					Arguments: marshalPlanTasksOrFail(t, []agent.PlanTask{
						{Content: "Research the topic", Status: agent.PlanTaskCompleted},
						{Content: "Synthesize findings", Status: agent.PlanTaskInProgress},
						{Content: "Draft comprehensive answer", Status: agent.PlanTaskPending},
					}),
				},
			}},
			// Step 4: Final answer after second plan revision
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "The answer based on revised plan"},
			}},
		},
	}

	// Create the unified planning/reflection tracker.
	// This tracker should:
	// - Record that delegated work has insufficient progress
	// - Emit planning/reflection events when revision is needed
	// - Signal that the plan must be revised
	tracker := agent.NewPlanningReflectionTracker()

	// Create the planning executor to track plan revisions
	planningExecutor := agent.NewPlanningExecutor(nil)

	// Create a unified policy that:
	// - Rejects answer when planning/reflection signals plan needs revision
	// - Accepts answer only after plan has been revised
	policy := agent.NewPlanningReflectionPolicy(
		planningExecutor,
		tracker,
		2, // Require at least 2 plan revisions for policy to accept
	)

	// Run the agent with the unified planning/reflection tracking
	result, events, err := agent.New(
		llm,
		[]model.ToolDefinition{agent.PlanningToolDefinition()},
		planningExecutor,
	).
		WithBeforeToolCallbacks(tracker).
		WithAfterToolCallbacks(tracker).
		WithFinalAnswerCallbacks(policy).
		WithMaxSteps(5).
		Run(context.Background(), "test topic")

	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Assert: Two planning revisions exist
	revisionEvents := collectPlanRevisionEvents(t, events)
	if len(revisionEvents) != 2 {
		t.Fatalf("len(revisionEvents) = %d, want 2", len(revisionEvents))
	}
	if revisionEvents[0].Revision.Index != 0 {
		t.Fatalf("first revision index = %d, want 0", revisionEvents[0].Revision.Index)
	}
	if revisionEvents[1].Revision.Index != 1 {
		t.Fatalf("second revision index = %d, want 1", revisionEvents[1].Revision.Index)
	}
	if revisionEvents[1].Revision.TaskCount != 3 {
		t.Fatalf("second revision task count = %d, want 3", revisionEvents[1].Revision.TaskCount)
	}

	// Assert: Planning/reflection events are recorded
	// These events should indicate that plan revision was triggered and recorded
	planReflectionEvents := collectPlanningReflectionEvents(t, events)
	if len(planReflectionEvents) < 1 {
		t.Fatalf("len(planReflectionEvents) = %d, want >= 1", len(planReflectionEvents))
	}

	// Assert: First answer is rejected with the new policy
	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) < 2 {
		t.Fatalf("len(policyEvents) = %d, want >= 2", len(policyEvents))
	}
	if policyEvents[0].Decision != agent.PolicyDecisionReject {
		t.Fatalf("first policy decision = %q, want reject", policyEvents[0].Decision)
	}
	if policyEvents[0].Answer != "The answer based on initial plan" {
		t.Fatalf("first rejected answer = %q", policyEvents[0].Answer)
	}

	// The second answer should be accepted only after the revised plan
	if policyEvents[1].Decision != agent.PolicyDecisionAccept {
		t.Fatalf("second policy decision = %q, want accept", policyEvents[1].Decision)
	}
	if policyEvents[1].Answer != "The answer based on revised plan" {
		t.Fatalf("accepted answer = %q", policyEvents[1].Answer)
	}

	// Assert: Final result contains the answer from the revised plan
	if result.Output != "The answer based on revised plan" {
		t.Fatalf("final output = %q, want revised answer", result.Output)
	}

	// Assert: Deferred user correction is appended after the triggering step
	// (The policy rejection message should guide the agent toward revision)
	contextEvents := result.Context.Events()
	if len(contextEvents) == 0 {
		t.Fatal("expected execution context events")
	}

	// Verify that we have captured the revised flow through context events
	var foundSecondPlan, foundFinalAnswer bool
	for _, evt := range contextEvents {
		for _, item := range evt.Content {
			if msg, ok := item.(model.Message); ok {
				if msg.Content == "The answer based on revised plan" {
					foundFinalAnswer = true
				}
			}
			if tc, ok := item.(model.ToolCall); ok {
				if tc.ID == "tc-plan-2" {
					foundSecondPlan = true
				}
			}
		}
	}

	if !foundSecondPlan {
		t.Fatal("expected second plan revision in context events")
	}
	if !foundFinalAnswer {
		t.Fatal("expected final answer in context events")
	}
}

func TestBeforeToolCallback_CanQueueDeferredUserMessage(t *testing.T) {
	t.Parallel()

	const prompt = "Finish the blocked tool step before continuing."
	llm := &deferredPromptLLM{}
	callback := beforeToolCallbackFunc(func(_ context.Context, execCtx *agent.ExecutionContext, call model.ToolCall) (*model.ToolResult, error) {
		agent.QueueDeferredUserMessage(execCtx, prompt)
		return &model.ToolResult{
			ID:      call.ID,
			Name:    call.Name,
			Status:  "blocked",
			Content: []string{"waiting for prerequisite"},
		}, nil
	})

	result, _, err := agent.New(
		llm,
		[]model.ToolDefinition{{Name: "gather_fact"}},
		nil,
	).WithBeforeToolCallbacks(callback).WithMaxSteps(3).Run(context.Background(), "test")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Output != "Prompt ordering stayed valid." {
		t.Fatalf("output = %q", result.Output)
	}
	if !hasUserMessageAfterBlockedResult(result.Context.Events(), prompt) {
		t.Fatal("expected deferred user prompt after blocked tool result")
	}
}

func TestPlanningReflection_RequiresReflectionAfterPlanningStagnation(t *testing.T) {
	t.Parallel()

	llm := &planningStagnationLLM{}
	tracker := agent.NewPlanningReflectionTracker(
		agent.WithPlanningReflectionStagnationThreshold(2),
	)
	planningExecutor := agent.NewPlanningExecutor(nil)
	policy := agent.NewPlanningReflectionPolicy(planningExecutor, tracker, 2)

	result, events, err := agent.New(
		llm,
		[]model.ToolDefinition{agent.PlanningToolDefinition()},
		planningExecutor,
	).
		WithBeforeToolCallbacks(tracker).
		WithAfterToolCallbacks(tracker).
		WithFinalAnswerCallbacks(policy).
		WithMaxSteps(6).
		Run(context.Background(), "test stagnation topic")

	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if result.Output == "" {
		t.Fatal("expected final output")
	}
	if tracker.LatestReflection() == "" {
		t.Fatal("expected recorded reflection after stagnation")
	}
	if !hasUserMessageAfterNthToolResult(
		result.Context.Events(),
		agent.PlanningToolDefinition().Name,
		"success",
		2,
		"Do not revise the plan again yet. First reply with a short reflection on why the plan is stalling and what you will change.",
	) {
		t.Fatal("expected stagnation correction after second planning result")
	}
	if !hasUserMessage(
		result.Context.Events(),
		"Thanks. Now revise your task list with create_tasks based on that reflection before answering.",
	) {
		t.Fatal("expected reflection follow-up revision prompt")
	}

	planReflectionEvents := collectPlanningReflectionEvents(t, events)
	if len(planReflectionEvents) < 3 {
		t.Fatalf("len(planReflectionEvents) = %d, want >= 3", len(planReflectionEvents))
	}
	if planReflectionEvents[0].Kind != agent.PlanningReflectionEventStagnationObserved {
		t.Fatalf("first planning reflection event = %q, want %q", planReflectionEvents[0].Kind, agent.PlanningReflectionEventStagnationObserved)
	}
	if planReflectionEvents[1].Kind != agent.PlanningReflectionEventReflectionRecorded {
		t.Fatalf("second planning reflection event = %q, want %q", planReflectionEvents[1].Kind, agent.PlanningReflectionEventReflectionRecorded)
	}
	if planReflectionEvents[len(planReflectionEvents)-1].Kind != agent.PlanningReflectionEventRevisionResolved {
		t.Fatalf("last planning reflection event = %q, want %q", planReflectionEvents[len(planReflectionEvents)-1].Kind, agent.PlanningReflectionEventRevisionResolved)
	}
}

func TestPlanningVerification_BlocksUnverifiedFinalAnswer(t *testing.T) {
	t.Parallel()

	llm := &verificationGateLLM{}
	collector := agent.NewEvidenceTracker(func(result model.ToolResult) (agent.EvidenceItem, bool) {
		if result.Name != "gather_fact" || result.Status != "success" || len(result.Content) == 0 {
			return agent.EvidenceItem{}, false
		}
		return agent.EvidenceItem{
			Source:  result.Name,
			Content: result.Content[0],
			Score:   1,
		}, true
	})
	gate := agent.NewVerificationGate(collector, 2)
	executor := &sequentialToolExecutor{
		batches: [][]model.ToolResult{
			{{ID: "tc-verify-1", Name: "gather_fact", Status: "success", Content: []string{"Record time: 2:01:09"}}},
			{{ID: "tc-verify-2", Name: "gather_fact", Status: "success", Content: []string{"Marathon distance: 42.195 km"}}},
		},
	}

	result, events, err := agent.New(
		llm,
		[]model.ToolDefinition{{Name: "gather_fact", Description: "Gather a supporting fact."}},
		executor,
	).
		WithAfterToolCallbacks(collector).
		WithFinalAnswerCallbacks(gate).
		WithMaxSteps(5).
		Run(context.Background(), "Verify Kipchoge's pace")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Output != "With two supporting facts gathered, Kipchoge's pace is about 20.9 km/h." {
		t.Fatalf("unexpected final output: %q", result.Output)
	}
	if got := len(collector.Evidence()); got != 2 {
		t.Fatalf("len(Evidence()) = %d, want 2", got)
	}

	policyEvents := collectPolicyEvents(t, events)
	if len(policyEvents) < 2 {
		t.Fatalf("len(policyEvents) = %d, want >= 2", len(policyEvents))
	}
	if policyEvents[0].Decision != agent.PolicyDecisionReject {
		t.Fatalf("first policy decision = %q, want reject", policyEvents[0].Decision)
	}
	if policyEvents[0].Reason != "Do not answer yet. Evidence is incomplete. Gather at least 2 supporting results before answering." {
		t.Fatalf("unexpected rejection reason: %q", policyEvents[0].Reason)
	}
	if policyEvents[len(policyEvents)-1].Decision != agent.PolicyDecisionAccept {
		t.Fatalf("last policy decision = %q, want accept", policyEvents[len(policyEvents)-1].Decision)
	}
}

// Helper function to marshal plan tasks or fail the test
func marshalPlanTasksOrFail(t *testing.T, tasks []agent.PlanTask) json.RawMessage {
	if t != nil {
		t.Helper()
	}
	result, err := agent.MarshalPlanTasks(tasks)
	if err != nil {
		if t != nil {
			t.Fatalf("MarshalPlanTasks() error = %v", err)
		}
		panic(err)
	}
	return result
}

// Helper function to collect planning/reflection events
func collectPlanningReflectionEvents(t *testing.T, events rxgo.Observable) []agent.PlanningReflectionEvent {
	t.Helper()
	var out []agent.PlanningReflectionEvent
	if events == nil {
		return out
	}
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.PlanningReflectionEvent); ok {
			out = append(out, event)
		}
	}
	return out
}
