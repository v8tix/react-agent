package agent

import (
	"context"
	"log/slog"
	"sync"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestWithMutatorLogger_LogsDelegateLifecycle(t *testing.T) {
	handler := newRecordingHandler()
	logger := slog.New(handler)
	called := false
	mutator := WithMutatorLogger(requestMutatorFunc(func(_ context.Context, req *model.Request) error {
		called = true
		req.Instructions += "\nlogged"
		return nil
	}), logger)
	req := model.Request{Instructions: "system"}

	if err := mutator.Mutate(context.Background(), &req); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("expected wrapped mutator to be called")
	}
	handler.assertMessages(t, "mutator_start", "mutator_finish")
}

func TestContextOptimizer_WithLogger_LogsOptimization(t *testing.T) {
	handler := newRecordingHandler()
	logger := slog.New(handler)
	optimizer := NewContextOptimizer(
		tokenCounterFunc(func(req model.Request) (int, error) {
			return len(req.Events), nil
		}),
		1,
		optimizationStrategyFunc(func(_ context.Context, req *model.Request) error {
			req.Events = req.Events[:1]
			return nil
		}),
	).WithLogger(logger)
	req := model.Request{
		Events: []model.Event{
			{Author: "user", Content: []model.ContentItem{model.Message{Role: "user", Content: "one"}}},
			{Author: "agent", Content: []model.ContentItem{model.Message{Role: "assistant", Content: "two"}}},
		},
	}

	if err := optimizer.Mutate(context.Background(), &req); err != nil {
		t.Fatal(err)
	}
	handler.assertMessages(t, "context_optimize_start", "context_strategy_apply", "context_strategy_applied")
}

func TestSessionRunner_WithLogger_LogsRunLifecycle(t *testing.T) {
	handler := newRecordingHandler()
	logger := slog.New(handler)
	llm := scriptedLLM(func(_ context.Context, _ model.Request) (model.Response, error) {
		return assistantResponse("ok"), nil
	})
	runner := NewSessionRunner(New(llm, nil, nil).WithMaxSteps(2), NewInMemorySessionManager(), 2).WithLogger(logger)

	result, err := runner.Run(context.Background(), "s1", "u1", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "ok" {
		t.Fatalf("unexpected output: %q", result.Output)
	}
	handler.assertMessages(t, "session_run_start", "session_run_end")
}

func TestConfirmationCallback_WithLogger_LogsApprovalRequest(t *testing.T) {
	handler := newRecordingHandler()
	logger := slog.New(handler)
	callback := NewConfirmationCallback(StaticApprovalPolicy{
		"delete_file": {MessageTemplate: "Approve?"},
	}).WithLogger(logger)
	execCtx := NewExecutionContextForTest()

	_, err := callback.BeforeTool(context.Background(), execCtx, model.ToolCall{ID: "tc-1", Name: "delete_file"})
	if err == nil {
		t.Fatal("expected suspend error")
	}
	handler.assertMessages(t, "approval_requested")
}

func TestTaskMemoryManager_WithLogger_LogsSaveAndSearch(t *testing.T) {
	handler := newRecordingHandler()
	logger := slog.New(handler)
	manager := NewTaskMemoryManager(
		staticEmbedder{
			"Task: find capitals": {1, 0},
			"find capitals":       {1, 0},
		},
		NewInMemoryVectorStore(),
		SimpleDuplicateChecker{},
	).WithLogger(logger)

	if _, saved, err := manager.Save(context.Background(), TaskMemory{
		TaskSummary: "find capitals",
		Approach:    "used search",
		FinalAnswer: "Paris",
		IsCorrect:   true,
	}); err != nil || !saved {
		t.Fatalf("unexpected save result: saved=%v err=%v", saved, err)
	}
	if _, err := manager.Search(context.Background(), "find capitals", 3); err != nil {
		t.Fatal(err)
	}
	handler.assertMessages(t, "memory_save_start", "memory_save_end", "memory_search_start", "memory_search_end")
}

type tokenCounterFunc func(model.Request) (int, error)

func (f tokenCounterFunc) Count(req model.Request) (int, error) { return f(req) }

type optimizationStrategyFunc func(context.Context, *model.Request) error

func (f optimizationStrategyFunc) Optimize(ctx context.Context, req *model.Request) error {
	return f(ctx, req)
}

type recordingHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func newRecordingHandler() *recordingHandler { return &recordingHandler{} }

func (h *recordingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *recordingHandler) Handle(_ context.Context, record slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, record.Clone())
	return nil
}

func (h *recordingHandler) WithAttrs(_ []slog.Attr) slog.Handler { return h }

func (h *recordingHandler) WithGroup(_ string) slog.Handler { return h }

func (h *recordingHandler) assertMessages(t *testing.T, want ...string) {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	got := make([]string, 0, len(h.records))
	for _, record := range h.records {
		got = append(got, record.Message)
	}
	for _, target := range want {
		found := false
		for _, message := range got {
			if message == target {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing log message %q in %#v", target, got)
		}
	}
}
