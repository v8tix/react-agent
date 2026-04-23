package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	agent "github.com/v8tix/react-agent"
	"github.com/v8tix/react-agent/model"
)

// ─── deterministic stubs used by all examples ────────────────────────────────

// demoLLM is a scripted LLM stub: it returns responses[0], responses[1], …
// in sequence. Use it to write deterministic, self-contained documentation
// examples that produce a known output without any network calls.
type demoLLM struct {
	responses []model.Response
	n         int
}

func (d *demoLLM) Generate(_ context.Context, _ model.Request) (model.Response, error) {
	resp := d.responses[d.n]
	d.n++
	return resp, nil
}

// demoExecutor echoes back a canned result for every tool call it receives.
// Suitable for examples that need a tool round-trip without a real backend.
type demoExecutor struct{}

func (demoExecutor) Execute(_ context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
	results := make([]model.ToolResult, len(calls))
	for i, c := range calls {
		results[i] = model.ToolResult{
			ID:      c.ID,
			Name:    c.Name,
			Status:  "success",
			Content: []string{fmt.Sprintf("result_of_%s", c.Name)},
		}
	}
	return results, nil
}

// ─── examples ────────────────────────────────────────────────────────────────

// ExampleNew shows how to construct an agent with the fluent builder.
// Replace demoLLM with agent.NewLiteLLMClient(openaiClient, model) to target
// a real LLM.
func ExampleNew() {
	// Hypothetical: ask an assistant to look up a stock price.
	llm := &demoLLM{} // swap for agent.NewLiteLLMClient(...)

	defs := []model.ToolDefinition{
		{
			Name:        "search_web",
			Description: "Search the web for up-to-date information",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string"},
				},
				"required": []string{"query"},
			},
		},
	}

	_ = agent.New(llm, defs, demoExecutor{}).
		WithInstructions("You are a precise research assistant.").
		WithMaxSteps(15)
}

// ExampleAgent_Run demonstrates a single-step run where the LLM answers
// directly without calling any tools.
func ExampleAgent_Run() {
	llm := &demoLLM{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "The capital of France is Paris."},
			}},
		},
	}

	a := agent.New(llm, nil, nil).
		WithInstructions("You are a helpful assistant.")

	result, _, err := a.Run(context.Background(), "What is the capital of France?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Output)
	fmt.Println(result.ToolCalled)
	// Output:
	// The capital of France is Paris.
	// false
}

// ExampleAgent_Run_withTools demonstrates a two-step run: the LLM first calls
// a tool, then produces a final answer once it has the search result.
//
// This mirrors the classic ReAct scenario — the agent reasons about what
// information it needs, fetches it, then synthesises an answer.
func ExampleAgent_Run_withTools() {
	llm := &demoLLM{
		responses: []model.Response{
			// Step 1: LLM decides to call search_web
			{Content: []model.ContentItem{
				model.ToolCall{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"AAPL stock price January 9 2007"}`),
				},
			}},
			// Step 2: LLM reads the search result and gives the final answer
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Apple stock was $11.74 on January 9, 2007."},
			}},
		},
	}

	defs := []model.ToolDefinition{{
		Name:        "search_web",
		Description: "Search the web for current information",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"query"},
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		},
	}}

	a := agent.New(llm, defs, demoExecutor{}).
		WithInstructions("You are a research assistant. Verify facts before answering.")

	result, _, err := a.Run(context.Background(), "What was Apple's stock price the day the iPhone was announced?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Output)
	fmt.Println("tool called:", result.ToolCalled)
	// Output:
	// Apple stock was $11.74 on January 9, 2007.
	// tool called: true
}

// ExampleAgent_Run_eventStream shows how to consume the observable event
// stream returned by Run to build logging, metrics, or tracing.
//
// The observable is cold and replayable — calling Observe() again replays
// all events from the beginning, safe for multiple independent subscribers.
func ExampleAgent_Run_eventStream() {
	llm := &demoLLM{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "c1", Name: "search_web", Arguments: json.RawMessage(`{}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Done."},
			}},
		},
	}

	defs := []model.ToolDefinition{{
		Name:       "search_web",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}, "required": []string{"query"}},
	}}

	a := agent.New(llm, defs, demoExecutor{})

	_, events, err := a.Run(context.Background(), "What is the weather in Paris?")
	if err != nil {
		log.Fatal(err)
	}

	for item := range events.Observe() {
		switch e := item.V.(type) {
		case agent.RunStartEvent:
			fmt.Println("run started")
		case agent.LLMCallEvent:
			fmt.Printf("llm call step=%d\n", e.Step)
		case agent.ToolExecEvent:
			fmt.Printf("tool exec: %v\n", e.ToolNames)
		case agent.RunEndEvent:
			fmt.Println("run ended")
		}
	}
	// Output:
	// run started
	// llm call step=0
	// tool exec: [search_web]
	// llm call step=1
	// run ended
}

// ExampleAgent_Step shows how to drive the ReAct loop manually step-by-step.
// This gives you control between steps — useful for streaming output to a UI,
// checkpointing long runs, or pausing for human approval before the agent acts.
func ExampleAgent_Step() {
	llm := &demoLLM{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "The answer is 42."},
			}},
		},
	}

	a := agent.New(llm, nil, nil).WithMaxSteps(10)

	execCtx := agent.NewExecutionContextForTest()
	execCtx.AddEvent("user", model.Message{Role: "user", Content: "What is the answer to life, the universe and everything?"})

	for execCtx.CurrentStep() < 10 {
		if err := a.Step(context.Background(), execCtx); err != nil {
			log.Fatal(err)
		}
		if execCtx.Done() {
			break
		}
		execCtx.IncrementStep()
	}

	answer, _ := execCtx.FinalResult()
	fmt.Println(answer)
	fmt.Println("done:", execCtx.Done())
	// Output:
	// The answer is 42.
	// done: true
}

// ExampleAgent_Run_reasoningTrail shows how to inspect the full conversation
// history — every message, tool call, and tool result — after a run.
// Useful for debugging, audit logs, or displaying the chain of thought.
func ExampleAgent_Run_reasoningTrail() {
	llm := &demoLLM{
		responses: []model.Response{
			{Content: []model.ContentItem{
				model.ToolCall{ID: "c1", Name: "lookup", Arguments: json.RawMessage(`{"id":"42"}`)},
			}},
			{Content: []model.ContentItem{
				model.Message{Role: "assistant", Content: "Found it."},
			}},
		},
	}

	defs := []model.ToolDefinition{{
		Name:       "lookup",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "string"}}, "required": []string{"id"}},
	}}

	result, _, err := agent.New(llm, defs, demoExecutor{}).
		Run(context.Background(), "Look up record 42.")
	if err != nil {
		log.Fatal(err)
	}

	for _, event := range result.Context.Events() {
		for _, item := range event.Content {
			switch v := item.(type) {
			case model.Message:
				fmt.Printf("[%s] %s\n", event.Author, v.Content)
			case model.ToolCall:
				fmt.Printf("[%s] call %s\n", event.Author, v.Name)
			case model.ToolResult:
				fmt.Printf("[%s] result %s=%s\n", event.Author, v.Name, v.Content[0])
			}
		}
	}
	// Output:
	// [user] Look up record 42.
	// [agent] call lookup
	// [tools] result lookup=result_of_lookup
	// [agent] Found it.
}
