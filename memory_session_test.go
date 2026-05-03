package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestInMemorySessionManager_GetOrCreateRejectsEmptyUserID(t *testing.T) {
	manager := NewInMemorySessionManager()
	if _, err := manager.GetOrCreate("s1", ""); err == nil {
		t.Fatal("want error for empty userID")
	}
}

func TestSessionRunner_RunCarriesConversationAcrossCalls(t *testing.T) {
	llm := scriptedLLM(func(_ context.Context, req model.Request) (model.Response, error) {
		switch lastUserMessage(req.Events) {
		case "My name is Alice":
			return assistantResponse("Nice to meet you, Alice!"), nil
		case "What's my name?":
			if containsMessage(req.Events, "My name is Alice") {
				return assistantResponse("Your name is Alice."), nil
			}
		}
		return assistantResponse("I don't know."), nil
	})
	runner := NewSessionRunner(New(llm, nil, nil).WithMaxSteps(4), NewInMemorySessionManager(), 4)

	first, err := runner.Run(context.Background(), "s1", "u1", "My name is Alice")
	if err != nil {
		t.Fatal(err)
	}
	second, err := runner.Run(context.Background(), "s1", "u1", "What's my name?")
	if err != nil {
		t.Fatal(err)
	}
	if first.Output != "Nice to meet you, Alice!" || second.Output != "Your name is Alice." {
		t.Fatalf("unexpected outputs: %#v %#v", first, second)
	}
}

func TestConfirmationCallback_RedactsPayloadAndResumes(t *testing.T) {
	args, _ := json.Marshal(map[string]any{"filename": "/tmp/secret.txt", "api_key": "secret"})
	llm := &sequenceLLM{responses: []model.Response{
		{Content: []model.ContentItem{model.ToolCall{ID: "tc-1", Name: "delete_file", Arguments: args}}},
		assistantResponse("Deleted temp.txt."),
	}}
	exec := &sessionStubToolExecutor{results: []model.ToolResult{{ID: "tc-1", Name: "delete_file", Status: "success", Content: []string{"Deleted temp.txt"}}}}
	runner := NewSessionRunner(
		New(llm, []model.ToolDefinition{{Name: "delete_file"}}, exec).
			WithBeforeToolCallbacks(NewConfirmationCallback(StaticApprovalPolicy{
				"delete_file": {MessageTemplate: "Approve?"},
			})).
			WithMaxSteps(4),
		NewInMemorySessionManager(),
		4,
	)

	pending, err := runner.Run(context.Background(), "s1", "u1", "delete temp.txt")
	if err != nil {
		t.Fatal(err)
	}
	payload := pending.PendingInteraction.Payload["arguments"].(map[string]any)
	if payload["api_key"] != "[redacted]" {
		t.Fatalf("api_key not redacted: %#v", payload["api_key"])
	}
	approved := true
	resumed, err := runner.Resume(context.Background(), "s1", "u1", InteractionResponse{RequestID: pending.PendingInteraction.ID, Approved: &approved})
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Output != "Deleted temp.txt." {
		t.Fatalf("unexpected resumed output: %q", resumed.Output)
	}
}

func TestSessionRunner_ResumesPersistedApprovalState(t *testing.T) {
	args, _ := json.Marshal(map[string]any{"filename": "/tmp/secret.txt"})
	llm := &sequenceLLM{responses: []model.Response{
		{Content: []model.ContentItem{model.ToolCall{ID: "tc-1", Name: "delete_file", Arguments: args}}},
		assistantResponse("Deleted temp.txt."),
	}}
	exec := &sessionStubToolExecutor{results: []model.ToolResult{{ID: "tc-1", Name: "delete_file", Status: "success", Content: []string{"Deleted temp.txt"}}}}
	persister := newInMemorySessionPersister()
	runner := NewSessionRunner(
		New(llm, []model.ToolDefinition{{Name: "delete_file"}}, exec).
			WithBeforeToolCallbacks(NewConfirmationCallback(StaticApprovalPolicy{
				"delete_file": {MessageTemplate: "Approve?"},
			})).
			WithMaxSteps(4),
		NewPersistedSessionManager(persister),
		4,
	)

	pending, err := runner.Run(context.Background(), "s1", "u1", "delete temp.txt")
	if err != nil {
		t.Fatal(err)
	}
	if pending.Status != StatusPending {
		t.Fatalf("status = %q, want pending", pending.Status)
	}
	approved := true
	resumed, err := runner.Resume(context.Background(), "s1", "u1", InteractionResponse{RequestID: pending.PendingInteraction.ID, Approved: &approved})
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Output != "Deleted temp.txt." {
		t.Fatalf("unexpected resumed output: %q", resumed.Output)
	}
}

type scriptedLLM func(context.Context, model.Request) (model.Response, error)

func (s scriptedLLM) Generate(ctx context.Context, req model.Request) (model.Response, error) {
	return s(ctx, req)
}

type sequenceLLM struct {
	responses []model.Response
	index     int
}

func (s *sequenceLLM) Generate(_ context.Context, _ model.Request) (model.Response, error) {
	if len(s.responses) == 0 {
		return model.Response{}, errors.New("no responses configured")
	}
	resp := s.responses[s.index]
	if s.index < len(s.responses)-1 {
		s.index++
	}
	return resp, nil
}

type sessionStubToolExecutor struct {
	results []model.ToolResult
}

func (s *sessionStubToolExecutor) Execute(_ context.Context, _ []model.ToolCall) ([]model.ToolResult, error) {
	return s.results, nil
}

func assistantResponse(text string) model.Response {
	return model.Response{Content: []model.ContentItem{model.Message{Role: "assistant", Content: text}}}
}

func containsMessage(events []model.Event, want string) bool {
	for _, event := range events {
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Content == want {
				return true
			}
		}
	}
	return false
}
