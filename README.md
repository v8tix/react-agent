# react-agent

**Build AI agents that think before they act.**

Most LLM integrations are one-shot: send a prompt, get an answer. But complex questions require *reasoning* — looking things up, checking results, adjusting the plan. `react-agent` implements the **ReAct pattern** (Reason + Act), a technique where the model alternates between *thinking out loud* and *using tools* until it's confident enough to answer.

> *Based on ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) — Yao et al., 2022*

---

## The ReAct pattern in plain language

Imagine asking an agent *"What was the stock price of Apple the day the iPhone was announced?"*

A one-shot model will guess. A ReAct agent will:

1. **Think** — *"I need to find the date of the first iPhone announcement"*
2. **Act** — calls `search_web("first iPhone announcement date")`
3. **Observe** — *"January 9, 2007"*
4. **Think** — *"Now I need Apple's stock price on that date"*
5. **Act** — calls `search_web("AAPL stock price January 9 2007")`
6. **Observe** — *"$11.74"*
7. **Answer** — *"Apple's stock was $11.74 on January 9, 2007, the day Steve Jobs unveiled the iPhone."*

That loop — think, act, observe, repeat — is what this library provides.

---

## Installation

```bash
go get github.com/v8tix/react-agent
```

---

## Quick start

```go
import agent "github.com/v8tix/react-agent"

// 1. Wrap your LLM (OpenAI or LiteLLM proxy)
client := agent.NewLiteLLMClient(openaiClient, "gpt-4o-mini")

// 2. Define your tools
defs := []agent.ToolDefinition{
    {
        Name:        "search_web",
        Description: "Search the web for current information",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "query": map[string]any{"type": "string"},
            },
            "required": []string{"query"},
        },
    },
}

// 3. Implement ToolExecutor (your dispatch logic)
executor := myToolExecutor

// 4. Create and run the agent
a := agent.New(client, defs, executor,
    agent.WithInstructions("You are a helpful assistant."),
    agent.WithMaxSteps(10))

result, err := a.Run(ctx, "Who won the 2025 Nobel Prize in Physics?")
fmt.Println(result.Output)
```

---

## Inspecting the reasoning trail

Every run captures the full event history in `result.Context`:

```go
for _, event := range result.Context.Events() {
    fmt.Printf("[%s] %s\n", event.Author, event.Timestamp.Format(time.RFC3339))
    for _, item := range event.Content {
        switch v := item.(type) {
        case agent.Message:
            fmt.Printf("  message: %s\n", v.Content)
        case agent.ToolCall:
            fmt.Printf("  tool_call: %s(%s)\n", v.Name, v.Arguments)
        case agent.ToolResult:
            fmt.Printf("  tool_result: [%s] %v\n", v.Status, v.Content)
        }
    }
}
```

---

## Implementing ToolExecutor

```go
type ToolExecutor interface {
    Execute(ctx context.Context, calls []agent.ToolCall) ([]agent.ToolResult, error)
}
```

A typical adapter looks like:

```go
type myExecutor struct{ /* your tool registry */ }

func (e *myExecutor) Execute(ctx context.Context, calls []agent.ToolCall) ([]agent.ToolResult, error) {
    results := make([]agent.ToolResult, len(calls))
    for i, call := range calls {
        output, err := e.run(ctx, call.Name, call.Arguments)
        if err != nil {
            results[i] = agent.ToolResult{ID: call.ID, Name: call.Name, Status: "error", Content: []string{err.Error()}}
            continue
        }
        results[i] = agent.ToolResult{ID: call.ID, Name: call.Name, Status: "success", Content: []string{output}}
    }
    return results, nil
}
```

---

## Driving the loop manually

`Step` is exported so you can control the loop yourself — useful for streaming, checkpointing, or human-in-the-loop interrupts:

```go
execCtx := agent.NewExecutionContextForTest() // or your own context
execCtx.AddEvent("user", agent.Message{Role: "user", Content: "hello"})

for execCtx.CurrentStep < 10 {
    if err := a.Step(ctx, execCtx); err != nil {
        break
    }
    // inspect execCtx.Events() between steps
    execCtx.IncrementStep()
}
```

---

## Design

| Type | Role |
|------|------|
| `Agent` | Orchestrator — owns the loop |
| `ExecutionContext` | Mutable run state — thread-safe event log |
| `Event` | Timestamped history entry (author + content items) |
| `LLMClient` | Interface — swap any provider |
| `ToolExecutor` | Interface — bring your own dispatch strategy |
| `LiteLLMClient` | Concrete adapter for openai-go / LiteLLM |

`agent/` imports only `github.com/openai/openai-go`. It has zero dependencies on any tool framework, making it easy to embed or extract.

---

## License

Apache 2.0
