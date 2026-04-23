// Package agent implements the ReAct (Reason + Act) pattern for AI agents.
//
// # Overview
//
// A ReAct agent runs a bounded Think → Act → Observe loop: the model thinks
// (generates a response), acts (calls tools), and observes (results are
// appended to the history), repeating until it produces a final answer or
// exhausts the step limit.
//
// The pattern is based on "ReAct: Synergizing Reasoning and Acting in
// Language Models" (Yao et al., 2022 — https://arxiv.org/abs/2210.03629).
//
// # Building an agent
//
// Use the fluent builder to compose an agent from an LLM client, tool
// definitions, and a tool executor:
//
//	a := agent.New(client, toolDefs, executor).
//	         WithInstructions("You are a precise research assistant.").
//	         WithMaxSteps(15)
//
// # Running an agent
//
// [Agent.Run] executes the full loop for a single user question. It returns a
// [Result], a replayable [rxgo.Observable] of [AgentEvent] values, and any error:
//
//	result, events, err := a.Run(ctx, "Who won the 2025 Nobel Prize in Physics?")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(result.Output)
//
// # Observable event stream
//
// The returned observable is a cold, replayable stream of everything that
// happened during the run. Subscribe by calling Observe():
//
//	for item := range events.Observe() {
//	    switch e := item.V.(type) {
//	    case agent.LLMCallEvent:
//	        slog.Info("llm call", "step", e.Step, "latency_ms", e.Latency.Milliseconds())
//	    case agent.ToolExecEvent:
//	        slog.Info("tool exec", "tools", e.ToolNames)
//	    case agent.RunEndEvent:
//	        slog.Info("run end", "err", e.Err)
//	    }
//	}
//
// Calling Observe() again replays all events from the beginning — safe for
// multiple independent subscribers (loggers, metrics, tracing).
//
// # Execution history
//
// The full conversation is available via result.Context.Events(). Each
// [model.Event] has an Author ("user", "agent", or "tools"), a timestamp, and
// typed [model.ContentItem] values ([model.Message], [model.ToolCall],
// [model.ToolResult]).
//
// # Bringing your own tools
//
// Implement [model.ToolExecutor] to connect any tool-running backend:
//
//	type myExecutor struct{ /* your registry, MCP session, etc. */ }
//
//	func (e *myExecutor) Execute(ctx context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
//	    results := make([]model.ToolResult, len(calls))
//	    for i, call := range calls {
//	        out, err := e.dispatch(ctx, call.Name, call.Arguments)
//	        if err != nil {
//	            results[i] = model.ToolResult{ID: call.ID, Name: call.Name, Status: "error", Content: []string{err.Error()}}
//	            continue
//	        }
//	        results[i] = model.ToolResult{ID: call.ID, Name: call.Name, Status: "success", Content: []string{out}}
//	    }
//	    return results, nil
//	}
//
// For MCP-based tools (github.com/v8tix/mcp-toolkit), use the ready-made
// adapter in the [mcpadapter] sub-package.
//
// # Manual step control
//
// [Agent.Step] is exported so callers can drive the loop themselves — useful
// for streaming, checkpointing, or human-in-the-loop interrupts:
//
//	execCtx := agent.NewExecutionContextForTest()
//	execCtx.AddEvent("user", model.Message{Role: "user", Content: question})
//
//	for execCtx.CurrentStep() < 20 {
//	    if err := a.Step(ctx, execCtx); err != nil {
//	        break
//	    }
//	    if execCtx.Done() {
//	        break
//	    }
//	    execCtx.IncrementStep()
//	}
package agent
