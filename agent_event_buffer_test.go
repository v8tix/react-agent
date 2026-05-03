package agent

import (
	"context"
	"testing"
	"time"

	"github.com/v8tix/react-agent/model"
)

type noisyAfterToolCallback struct {
	count int
}

func (c noisyAfterToolCallback) AfterTool(
	ctx context.Context,
	execCtx *ExecutionContext,
	result model.ToolResult,
) (*model.ToolResult, error) {
	ch, _ := ctx.Value(planningEventChannelKey{}).(chan<- AgentEvent)
	for i := 0; i < c.count; i++ {
		emit(ch, CallbackEvent{
			RunID:    execCtx.ID(),
			Step:     execCtx.CurrentStep(),
			Phase:    CallbackPhaseAfterTool,
			Stage:    CallbackStageFinish,
			Callback: "noisyAfterToolCallback",
			ToolName: result.Name,
		})
	}
	return nil, nil
}

type singleToolThenAnswerLLM struct {
	calls int
}

func (l *singleToolThenAnswerLLM) Generate(_ context.Context, _ model.Request) (model.Response, error) {
	if l.calls == 0 {
		l.calls++
		return model.Response{
			Content: []model.ContentItem{
				model.ToolCall{ID: "tc1", Name: "search"},
			},
		}, nil
	}
	return model.Response{
		Content: []model.ContentItem{
			model.Message{Role: "assistant", Content: "done"},
		},
	}, nil
}

type singleResultExecutor struct{}

func (singleResultExecutor) Execute(_ context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
	out := make([]model.ToolResult, len(calls))
	for i, call := range calls {
		out[i] = model.ToolResult{
			ID:      call.ID,
			Name:    call.Name,
			Status:  "success",
			Content: []string{"ok"},
		}
	}
	return out, nil
}

func TestAgentRun_CompletesWhenCallbacksEmitManyEvents(t *testing.T) {
	t.Parallel()

	a := New(
		&singleToolThenAnswerLLM{},
		[]model.ToolDefinition{{Name: "search"}},
		singleResultExecutor{},
	).WithAfterToolCallbacks(
		noisyAfterToolCallback{count: 100},
	).WithMaxSteps(2)

	done := make(chan error, 1)
	go func() {
		_, _, err := a.Run(context.Background(), "trigger")
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run() returned error: %v", err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Run() did not complete; likely blocked while emitting events")
	}
}

func TestStartEventCollector_ScalesBufferWithMaxSteps(t *testing.T) {
	t.Parallel()

	ch, wait := startEventCollector(30)
	if got, wantMin := cap(ch), 30*4+8; got < wantMin {
		t.Fatalf("collector buffer = %d, want at least %d", got, wantMin)
	}
	close(ch)
	if len(wait()) != 0 {
		t.Fatal("expected no collected events")
	}
}
