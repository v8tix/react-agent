package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/reactivex/rxgo/v2"
	agent "github.com/v8tix/react-agent"
	"github.com/v8tix/react-agent/model"
)

type stubBeforeToolCallback struct {
	result *model.ToolResult
	err    error
	calls  []model.ToolCall
}

func (s *stubBeforeToolCallback) BeforeTool(_ context.Context, _ *agent.ExecutionContext, call model.ToolCall) (*model.ToolResult, error) {
	s.calls = append(s.calls, call)
	return s.result, s.err
}

type stubAfterToolCallback struct {
	result  *model.ToolResult
	err     error
	results []model.ToolResult
}

func (s *stubAfterToolCallback) AfterTool(_ context.Context, _ *agent.ExecutionContext, result model.ToolResult) (*model.ToolResult, error) {
	s.results = append(s.results, result)
	return s.result, s.err
}

type selectiveBeforeToolCallback struct {
	name   string
	result *model.ToolResult
}

func (s selectiveBeforeToolCallback) BeforeTool(_ context.Context, _ *agent.ExecutionContext, call model.ToolCall) (*model.ToolResult, error) {
	if call.Name != s.name {
		return nil, nil
	}
	return s.result, nil
}

func collectCallbackEvents(t *testing.T, events rxgo.Observable) []agent.CallbackEvent {
	t.Helper()

	var out []agent.CallbackEvent
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.CallbackEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

func collectInteractionRequestedEvents(t *testing.T, events rxgo.Observable) []agent.InteractionRequestedEvent {
	t.Helper()
	var out []agent.InteractionRequestedEvent
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.InteractionRequestedEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

func collectInteractionResumedEvents(t *testing.T, events rxgo.Observable) []agent.InteractionResumedEvent {
	t.Helper()
	var out []agent.InteractionResumedEvent
	for item := range events.Observe() {
		if item.E != nil {
			t.Fatal(item.E)
		}
		if event, ok := item.V.(agent.InteractionResumedEvent); ok {
			out = append(out, event)
		}
	}
	return out
}

type interactiveApprovalBeforeCallback struct {
	deniedMessage string
}

func (c interactiveApprovalBeforeCallback) BeforeTool(_ context.Context, execCtx *agent.ExecutionContext, call model.ToolCall) (*model.ToolResult, error) {
	requestID := "approve-" + call.ID
	if resp, ok := execCtx.InteractionResponse(requestID); ok {
		if resp.Approved != nil && *resp.Approved {
			return nil, nil
		}
		return &model.ToolResult{Status: "error", Content: []string{c.deniedMessage}}, nil
	}
	return nil, agent.Suspend(agent.InteractionRequest{
		ID:         requestID,
		Kind:       "approval",
		Prompt:     "Approve dangerous tool?",
		ToolCallID: call.ID,
		ToolName:   call.Name,
	})
}

func TestAgent_Act_BeforeToolCallbackShortCircuitsExecution(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	before := selectiveBeforeToolCallback{
		name:   "search",
		result: &model.ToolResult{Status: "success", Content: []string{"cached result"}},
	}
	executor := &mockToolExecutor{}
	a := agent.New(&mockLLMClient{}, []model.ToolDefinition{{Name: "search"}}, executor).
		WithBeforeToolCallbacks(before)
	execCtx := agent.NewExecutionContextForTest()

	if err := a.Act(context.Background(), execCtx, []model.ToolCall{call}); err != nil {
		t.Fatal(err)
	}

	executor.mu.Lock()
	defer executor.mu.Unlock()
	if len(executor.calls) != 0 {
		t.Fatalf("executor should not have been called, got %d calls", len(executor.calls))
	}

	events := execCtx.Events()
	if len(events) != 2 {
		t.Fatalf("want 2 events, got %d", len(events))
	}
	got, ok := events[1].Content[0].(model.ToolResult)
	if !ok {
		t.Fatalf("want ToolResult, got %T", events[1].Content[0])
	}
	if got.Content[0] != "cached result" {
		t.Fatalf("tool result content = %q, want %q", got.Content[0], "cached result")
	}
	if got.ID != call.ID || got.Name != call.Name {
		t.Fatalf("tool result identity = (%q,%q), want (%q,%q)", got.ID, got.Name, call.ID, call.Name)
	}
}

func TestAgent_Act_AfterToolCallbackReplacesResult(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	after := &stubAfterToolCallback{
		result: &model.ToolResult{Status: "success", Content: []string{"compressed result"}},
	}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: call.ID, Name: call.Name, Status: "success", Content: []string{"original result"}}},
	}
	a := agent.New(&mockLLMClient{}, []model.ToolDefinition{{Name: "search"}}, executor).
		WithAfterToolCallbacks(after)
	execCtx := agent.NewExecutionContextForTest()

	if err := a.Act(context.Background(), execCtx, []model.ToolCall{call}); err != nil {
		t.Fatal(err)
	}

	events := execCtx.Events()
	if len(events) != 2 {
		t.Fatalf("want 2 events, got %d", len(events))
	}
	got, ok := events[1].Content[0].(model.ToolResult)
	if !ok {
		t.Fatalf("want ToolResult, got %T", events[1].Content[0])
	}
	if got.Content[0] != "compressed result" {
		t.Fatalf("tool result content = %q, want %q", got.Content[0], "compressed result")
	}
}

func TestAgent_Act_CallbacksUseEarlyExit(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	first := &stubBeforeToolCallback{
		result: &model.ToolResult{Status: "success", Content: []string{"first override"}},
	}
	second := &stubBeforeToolCallback{
		result: &model.ToolResult{Status: "success", Content: []string{"second override"}},
	}
	a := agent.New(&mockLLMClient{}, []model.ToolDefinition{{Name: "search"}}, &mockToolExecutor{}).
		WithBeforeToolCallbacks(first, second)
	execCtx := agent.NewExecutionContextForTest()

	if err := a.Act(context.Background(), execCtx, []model.ToolCall{call}); err != nil {
		t.Fatal(err)
	}

	if len(first.calls) != 1 {
		t.Fatalf("first callback calls = %d, want 1", len(first.calls))
	}
	if len(second.calls) != 0 {
		t.Fatalf("second callback calls = %d, want 0", len(second.calls))
	}
}

func TestAgent_Act_CallbackErrorsAbortExecution(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	sentinel := errors.New("approval callback failed")
	before := &stubBeforeToolCallback{err: sentinel}
	executor := &mockToolExecutor{}
	a := agent.New(&mockLLMClient{}, []model.ToolDefinition{{Name: "search"}}, executor).
		WithBeforeToolCallbacks(before)
	execCtx := agent.NewExecutionContextForTest()

	err := a.Act(context.Background(), execCtx, []model.ToolCall{call})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("want errors.Is(err, sentinel), got %v", err)
	}

	executor.mu.Lock()
	defer executor.mu.Unlock()
	if len(executor.calls) != 0 {
		t.Fatalf("executor should not have been called, got %d calls", len(executor.calls))
	}
}

func TestAgent_Run_EmitsBeforeToolCallbackEvent(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	before := &stubBeforeToolCallback{
		result: &model.ToolResult{Status: "success", Content: []string{"cached result"}},
	}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}

	result, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, &mockToolExecutor{}).
		WithBeforeToolCallbacks(before).
		Run(context.Background(), "find paris")
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("expected result")
	}

	callbackEvents := collectCallbackEvents(t, events)

	if len(callbackEvents) != 2 {
		t.Fatalf("want 2 callback events, got %d", len(callbackEvents))
	}
	if callbackEvents[0].Stage != agent.CallbackStageStart {
		t.Fatalf("start stage = %q, want %q", callbackEvents[0].Stage, agent.CallbackStageStart)
	}
	got := callbackEvents[1]
	if got.Phase != agent.CallbackPhaseBeforeTool {
		t.Fatalf("phase = %q, want %q", got.Phase, agent.CallbackPhaseBeforeTool)
	}
	if got.Stage != agent.CallbackStageFinish {
		t.Fatalf("finish stage = %q, want %q", got.Stage, agent.CallbackStageFinish)
	}
	if got.ToolName != "search" {
		t.Fatalf("tool name = %q, want %q", got.ToolName, "search")
	}
	if !got.Overrode {
		t.Fatal("expected override event")
	}
	if got.Err != nil {
		t.Fatalf("expected nil err, got %v", got.Err)
	}
}

func TestAgent_Run_EmitsAfterToolCallbackEvent(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	after := &stubAfterToolCallback{
		result: &model.ToolResult{Status: "success", Content: []string{"compressed result"}},
	}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: call.ID, Name: call.Name, Status: "success", Content: []string{"original result"}}},
	}

	result, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, executor).
		WithAfterToolCallbacks(after).
		Run(context.Background(), "find paris")
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("expected result")
	}

	callbackEvents := collectCallbackEvents(t, events)

	if len(callbackEvents) != 2 {
		t.Fatalf("want 2 callback events, got %d", len(callbackEvents))
	}
	if callbackEvents[0].Stage != agent.CallbackStageStart {
		t.Fatalf("start stage = %q, want %q", callbackEvents[0].Stage, agent.CallbackStageStart)
	}
	got := callbackEvents[1]
	if got.Phase != agent.CallbackPhaseAfterTool {
		t.Fatalf("phase = %q, want %q", got.Phase, agent.CallbackPhaseAfterTool)
	}
	if got.Stage != agent.CallbackStageFinish {
		t.Fatalf("finish stage = %q, want %q", got.Stage, agent.CallbackStageFinish)
	}
	if got.ToolName != "search" {
		t.Fatalf("tool name = %q, want %q", got.ToolName, "search")
	}
	if !got.Overrode {
		t.Fatal("expected override event")
	}
	if got.Err != nil {
		t.Fatalf("expected nil err, got %v", got.Err)
	}
}

func TestAgent_Run_EmitsBeforeToolCallbackEventWithoutOverride(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	before := &stubBeforeToolCallback{}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}

	_, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, &mockToolExecutor{}).
		WithBeforeToolCallbacks(before).
		Run(context.Background(), "find paris")
	if err != nil {
		t.Fatal(err)
	}

	callbackEvents := collectCallbackEvents(t, events)
	if len(callbackEvents) != 2 {
		t.Fatalf("want 2 callback events, got %d", len(callbackEvents))
	}
	if callbackEvents[0].Stage != agent.CallbackStageStart {
		t.Fatalf("start stage = %q, want %q", callbackEvents[0].Stage, agent.CallbackStageStart)
	}
	if callbackEvents[1].Overrode {
		t.Fatal("expected non-override callback finish event")
	}
	if callbackEvents[1].Phase != agent.CallbackPhaseBeforeTool {
		t.Fatalf("phase = %q, want %q", callbackEvents[1].Phase, agent.CallbackPhaseBeforeTool)
	}
}

func TestAgent_Run_EmitsAfterToolCallbackEventWithoutOverride(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	after := &stubAfterToolCallback{}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: call.ID, Name: call.Name, Status: "success", Content: []string{"original result"}}},
	}

	_, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, executor).
		WithAfterToolCallbacks(after).
		Run(context.Background(), "find paris")
	if err != nil {
		t.Fatal(err)
	}

	callbackEvents := collectCallbackEvents(t, events)
	if len(callbackEvents) != 2 {
		t.Fatalf("want 2 callback events, got %d", len(callbackEvents))
	}
	if callbackEvents[0].Stage != agent.CallbackStageStart {
		t.Fatalf("start stage = %q, want %q", callbackEvents[0].Stage, agent.CallbackStageStart)
	}
	if callbackEvents[1].Overrode {
		t.Fatal("expected non-override callback finish event")
	}
	if callbackEvents[1].Phase != agent.CallbackPhaseAfterTool {
		t.Fatalf("phase = %q, want %q", callbackEvents[1].Phase, agent.CallbackPhaseAfterTool)
	}
}

func TestAgent_Run_EmitsMultipleBeforeToolCallbackEventsInOrder(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	first := &stubBeforeToolCallback{}
	second := &stubBeforeToolCallback{
		result: &model.ToolResult{Status: "success", Content: []string{"cached result"}},
	}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}

	_, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, &mockToolExecutor{}).
		WithBeforeToolCallbacks(first, second).
		Run(context.Background(), "find paris")
	if err != nil {
		t.Fatal(err)
	}

	callbackEvents := collectCallbackEvents(t, events)
	if len(callbackEvents) != 4 {
		t.Fatalf("want 4 callback events, got %d", len(callbackEvents))
	}
	if callbackEvents[0].Stage != agent.CallbackStageStart || callbackEvents[1].Stage != agent.CallbackStageFinish {
		t.Fatal("first callback should emit start then finish")
	}
	if callbackEvents[1].Overrode {
		t.Fatal("first callback should not override")
	}
	if callbackEvents[2].Stage != agent.CallbackStageStart || callbackEvents[3].Stage != agent.CallbackStageFinish {
		t.Fatal("second callback should emit start then finish")
	}
	if !callbackEvents[3].Overrode {
		t.Fatal("second callback should override on finish")
	}
	if len(first.calls) != 1 || len(second.calls) != 1 {
		t.Fatalf("callback invocations = (%d,%d), want (1,1)", len(first.calls), len(second.calls))
	}
}

func TestAgent_Run_EmitsAfterToolCallbackErrorEvent(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	sentinel := errors.New("compression failed")
	after := &stubAfterToolCallback{err: sentinel}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
	}}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: call.ID, Name: call.Name, Status: "success", Content: []string{"original result"}}},
	}

	_, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, executor).
		WithAfterToolCallbacks(after).
		Run(context.Background(), "find paris")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("want errors.Is(err, sentinel), got %v", err)
	}

	callbackEvents := collectCallbackEvents(t, events)
	if len(callbackEvents) != 2 {
		t.Fatalf("want 2 callback events, got %d", len(callbackEvents))
	}
	if callbackEvents[0].Stage != agent.CallbackStageStart {
		t.Fatalf("start stage = %q, want %q", callbackEvents[0].Stage, agent.CallbackStageStart)
	}
	if !errors.Is(callbackEvents[1].Err, sentinel) {
		t.Fatalf("want callback event error %v, got %v", sentinel, callbackEvents[1].Err)
	}
	if callbackEvents[1].Phase != agent.CallbackPhaseAfterTool {
		t.Fatalf("phase = %q, want %q", callbackEvents[1].Phase, agent.CallbackPhaseAfterTool)
	}
	if callbackEvents[1].Stage != agent.CallbackStageFinish {
		t.Fatalf("finish stage = %q, want %q", callbackEvents[1].Stage, agent.CallbackStageFinish)
	}
}

func TestAgent_Act_MixedOverriddenAndExecutedToolCalls(t *testing.T) {
	t.Parallel()

	calls := []model.ToolCall{
		{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)},
		{ID: "tc2", Name: "lookup", Arguments: json.RawMessage(`{"id":"42"}`)},
	}
	before := selectiveBeforeToolCallback{
		name:   "search",
		result: &model.ToolResult{Status: "success", Content: []string{"cached result"}},
	}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: "tc2", Name: "lookup", Status: "success", Content: []string{"live result"}}},
	}
	a := agent.New(&mockLLMClient{}, []model.ToolDefinition{{Name: "search"}, {Name: "lookup"}}, executor).
		WithBeforeToolCallbacks(before)
	execCtx := agent.NewExecutionContextForTest()

	if err := a.Act(context.Background(), execCtx, calls); err != nil {
		t.Fatal(err)
	}

	executor.mu.Lock()
	if len(executor.calls) != 1 {
		t.Fatalf("executor calls = %d, want 1", len(executor.calls))
	}
	if executor.calls[0].Name != "lookup" {
		t.Fatalf("executor call name = %q, want %q", executor.calls[0].Name, "lookup")
	}
	executor.mu.Unlock()

	events := execCtx.Events()
	resultsEvent := events[len(events)-1]
	firstResult, ok := resultsEvent.Content[0].(model.ToolResult)
	if !ok {
		t.Fatalf("first result type = %T, want ToolResult", resultsEvent.Content[0])
	}
	secondResult, ok := resultsEvent.Content[1].(model.ToolResult)
	if !ok {
		t.Fatalf("second result type = %T, want ToolResult", resultsEvent.Content[1])
	}
	if firstResult.Content[0] != "cached result" {
		t.Fatalf("first result = %q, want %q", firstResult.Content[0], "cached result")
	}
	if secondResult.Content[0] != "live result" {
		t.Fatalf("second result = %q, want %q", secondResult.Content[0], "live result")
	}
}

func TestAgent_Run_NoCallbacksEmitsNoCallbackEvents(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Paris"}`)}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}

	_, events, err := agent.New(mock, []model.ToolDefinition{{Name: "search"}}, &mockToolExecutor{}).
		Run(context.Background(), "find paris")
	if err != nil {
		t.Fatal(err)
	}

	callbackEvents := collectCallbackEvents(t, events)
	if len(callbackEvents) != 0 {
		t.Fatalf("want 0 callback events, got %d", len(callbackEvents))
	}
}

func TestAgent_Run_SuspendsOnInteractionRequest(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "delete_file", Arguments: json.RawMessage(`{"path":"danger.txt"}`)}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
	}}
	executor := &mockToolExecutor{}

	result, events, err := agent.New(mock, []model.ToolDefinition{{Name: "delete_file"}}, executor).
		WithBeforeToolCallbacks(interactiveApprovalBeforeCallback{deniedMessage: "blocked"}).
		Run(context.Background(), "delete it")
	if result != nil {
		t.Fatal("expected nil result on suspension")
	}
	var suspendedErr *agent.InteractionRequestedError
	if !errors.As(err, &suspendedErr) {
		t.Fatalf("want InteractionRequestedError, got %v", err)
	}
	if suspendedErr.Suspended.Interaction.Kind != "approval" {
		t.Fatalf("interaction kind = %q, want %q", suspendedErr.Suspended.Interaction.Kind, "approval")
	}
	if _, ok := suspendedErr.Suspended.Context.PendingInteraction(); !ok {
		t.Fatal("expected pending interaction on execution context")
	}

	requested := collectInteractionRequestedEvents(t, events)
	if len(requested) != 1 {
		t.Fatalf("want 1 interaction request event, got %d", len(requested))
	}

	executor.mu.Lock()
	defer executor.mu.Unlock()
	if len(executor.calls) != 0 {
		t.Fatalf("executor should not have been called, got %d calls", len(executor.calls))
	}
}

func TestAgent_Resume_ApprovedContinuesExecution(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "delete_file", Arguments: json.RawMessage(`{"path":"danger.txt"}`)}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: call.ID, Name: call.Name, Status: "success", Content: []string{"deleted"}}},
	}
	a := agent.New(mock, []model.ToolDefinition{{Name: "delete_file"}}, executor).
		WithBeforeToolCallbacks(interactiveApprovalBeforeCallback{deniedMessage: "blocked"})

	_, _, err := a.Run(context.Background(), "delete it")
	var suspendedErr *agent.InteractionRequestedError
	if !errors.As(err, &suspendedErr) {
		t.Fatalf("want InteractionRequestedError, got %v", err)
	}

	approved := true
	result, events, err := a.Resume(context.Background(), suspendedErr.Suspended, agent.InteractionResponse{
		RequestID: suspendedErr.Suspended.Interaction.ID,
		Approved:  &approved,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || result.Output != "done" {
		t.Fatalf("result output = %#v, want done", result)
	}

	resumed := collectInteractionResumedEvents(t, events)
	if len(resumed) != 1 {
		t.Fatalf("want 1 interaction resumed event, got %d", len(resumed))
	}

	executor.mu.Lock()
	defer executor.mu.Unlock()
	if len(executor.calls) != 1 {
		t.Fatalf("executor calls = %d, want 1", len(executor.calls))
	}
}

func TestAgent_Resume_ObservableReplaysAcrossSubscriptions(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "delete_file", Arguments: json.RawMessage(`{"path":"danger.txt"}`)}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "done"}}},
	}}
	executor := &mockToolExecutor{
		results: []model.ToolResult{{ID: call.ID, Name: call.Name, Status: "success", Content: []string{"deleted"}}},
	}
	a := agent.New(mock, []model.ToolDefinition{{Name: "delete_file"}}, executor).
		WithBeforeToolCallbacks(interactiveApprovalBeforeCallback{deniedMessage: "blocked"})

	_, _, err := a.Run(context.Background(), "delete it")
	var suspendedErr *agent.InteractionRequestedError
	if !errors.As(err, &suspendedErr) {
		t.Fatalf("want InteractionRequestedError, got %v", err)
	}

	approved := true
	_, events, err := a.Resume(context.Background(), suspendedErr.Suspended, agent.InteractionResponse{
		RequestID: suspendedErr.Suspended.Interaction.ID,
		Approved:  &approved,
	})
	if err != nil {
		t.Fatal(err)
	}

	first := collectInteractionResumedEvents(t, events)
	second := collectInteractionResumedEvents(t, events)
	if len(first) != 1 || len(second) != 1 {
		t.Fatalf("resumed replay counts = %d and %d, want 1 each", len(first), len(second))
	}
}

func TestAgent_Resume_RejectedReturnsOverrideWithoutExecuting(t *testing.T) {
	t.Parallel()

	call := model.ToolCall{ID: "tc1", Name: "delete_file", Arguments: json.RawMessage(`{"path":"danger.txt"}`)}
	mock := &mockLLMClient{responses: []model.Response{
		{Content: []model.ContentItem{call}},
		{Content: []model.ContentItem{model.Message{Role: "assistant", Content: "blocked"}}},
	}}
	executor := &mockToolExecutor{}
	a := agent.New(mock, []model.ToolDefinition{{Name: "delete_file"}}, executor).
		WithBeforeToolCallbacks(interactiveApprovalBeforeCallback{deniedMessage: "blocked"})

	_, _, err := a.Run(context.Background(), "delete it")
	var suspendedErr *agent.InteractionRequestedError
	if !errors.As(err, &suspendedErr) {
		t.Fatalf("want InteractionRequestedError, got %v", err)
	}

	approved := false
	result, _, err := a.Resume(context.Background(), suspendedErr.Suspended, agent.InteractionResponse{
		RequestID: suspendedErr.Suspended.Interaction.ID,
		Approved:  &approved,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || result.Output != "blocked" {
		t.Fatalf("result output = %#v, want blocked", result)
	}

	executor.mu.Lock()
	defer executor.mu.Unlock()
	if len(executor.calls) != 0 {
		t.Fatalf("executor calls = %d, want 0", len(executor.calls))
	}
}
