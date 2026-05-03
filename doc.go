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
// When a workflow should expose only a subset of tools at a given step, add
// [Agent.WithDynamicToolsCallback]:
//
//	a := agent.New(client, toolDefs, executor).
//	         WithDynamicToolsCallback(func(execCtx *agent.ExecutionContext) []model.ToolDefinition {
//	             _ = execCtx // inspect current events or state here
//	             return toolDefs[:1]
//	         })
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
// If you want live logging while the run is still executing, attach a sink with
// [Agent.WithLiveEventSink]:
//
//	a := agent.New(client, toolDefs, executor).
//	         WithLiveEventSink(func(event agent.AgentEvent) {
//	             if e, ok := event.(agent.LLMCallEvent); ok {
//	                 slog.Info("live llm call", "step", e.Step, "latency_ms", e.Latency.Milliseconds())
//	             }
//	         })
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
// For MCP-based tools (github.com/v8tix/mcp-toolkit/v2), use the ready-made
// adapter in the [mcpadapter] sub-package.
//
// # Stateful sessions and approvals
//
// Use [SessionRunner] when a conversation must persist across multiple
// user-facing turns. It replays prior [model.Event] values from a
// [SessionManager], runs the agent, and saves the updated state after each call.
// If a callback suspends the run, [SessionRunner.Run] returns [StatusPending]
// plus a pending interaction payload that your app can surface in a UI or API
// before resuming. Use [NewPersistedSessionManager] with a [SessionPersister]
// when that state must survive process boundaries:
//
//	sessions := agent.NewInMemorySessionManager()
//	runner := agent.NewSessionRunner(
//	    agent.New(client, defs, executor).
//	        WithBeforeToolCallbacks(agent.NewConfirmationCallback(agent.StaticApprovalPolicy{
//	            "delete_file": {MessageTemplate: "Approve file deletion?"},
//	        })),
//	    sessions,
//	    8,
//	)
//
//	first, _ := runner.Run(ctx, "chat-1", "user-7", "My name is Alice")
//	next, _ := runner.Run(ctx, "chat-1", "user-7", "What's my name?")
//	_, _ = first, next
//
// Approval callbacks use [Suspend] under the hood and can be resumed with
// [Agent.Resume] or [SessionRunner.Resume]. The built-in [ConfirmationCallback]
// also redacts sensitive tool arguments from the interaction payload.
// Read and write [Session.State] when later turns depend on facts captured
// earlier — it works well as shared scratch space for multi-step workflows.
//
// # Planning and reflection policies
//
// For tasks that benefit from explicit planning, pair [NewPlanningExecutor]
// with [PlanningToolDefinition]. Add [NewPlanningReflectionTracker] plus
// [NewPlanningReflectionPolicy] when you want the agent to revise its task list
// before finalizing. Use [WithPlanningReflectionStagnationThreshold] to turn on
// a stricter loop where repeated planning-only churn triggers an explicit
// reflection step before more planning is allowed. When final answers must be
// grounded in gathered facts, add [NewVerificationGate] after an
// [EvidenceCollector] has started recording support for the answer.
//
// # Workflow-owned control
//
// Some applications need more than generic tool use. They need a bounded
// workflow where the model can still reason, but the application controls the
// critical phases. A friendly way to think about this is:
//
//	plan -> deterministic step -> fallback if needed -> gather evidence -> grounded answer
//
// `react-agent` keeps those workflow rules out of the core runtime. Instead it
// exposes small seams so the application can own the policy:
//
//   - [Agent.WithDynamicToolsCallback] can hide tools that should not be visible
//     in the current phase.
//   - [BeforeToolCallback] can block illegal tool choices, trigger circuit
//     breakers, or queue corrective user messages.
//   - [AfterToolCallback] can record state transitions after success or failure.
//   - [FinalAnswerCallback] can reject an answer that is not grounded in the
//     facts already gathered by the workflow.
//   - [QueueDeferredUserMessage] lets callbacks steer the next turn without
//     breaking the event ordering guarantees of the loop.
//
// A typical workflow-controlled setup looks like:
//
//	phaseTracker := newMyWorkflowTracker()
//	a := agent.New(client, defs, executor).
//	         WithDynamicToolsCallback(func(execCtx *agent.ExecutionContext) []model.ToolDefinition {
//	             return phaseTracker.AllowedTools(defs)
//	         }).
//	         WithBeforeToolCallbacks(phaseTracker).
//	         WithAfterToolCallbacks(phaseTracker).
//	         WithFinalAnswerCallbacks(myWorkflowGate{tracker: phaseTracker})
//
// In that pattern, the library still owns the ReAct loop, history, events, and
// suspension/resume flow, while the application owns the business-specific
// workflow phases.
//
// # Request mutation and context memory
//
// [MutatingLLMClient] lets you rewrite a request immediately before it is sent
// to the underlying [LLMClient]. This is the extension point for prompt
// hygiene, context-window management, and memory injection.
//
// Common building blocks:
//
//   - [ContextOptimizer] applies one or more [OptimizationStrategy] values once
//     a [TokenCounter] threshold is exceeded.
//   - [SlidingWindowStrategy] preserves the latest user turn and a recent tail of
//     events.
//   - [CompactionStrategy] replaces bulky tool payloads with short sanitized
//     summaries.
//   - [SummarizationStrategy] moves older history into a generated summary in
//     the instructions.
//   - [WithMutatorLogger] adds structured logs around any [RequestMutator].
//   - [StablePrefixDetector] is a small seam for apps that want to identify the
//     reusable prefix of a request for caching-friendly workflows. This is most
//     useful when your provider supports prompt caching and the request has a
//     large stable setup section.
//
// # Long-term task memory
//
// [TaskMemoryManager] stores solved tasks in a pluggable [VectorStore] so future
// requests can retrieve similar work. Pair it with [MemoryInjector] to inject
// the most relevant prior records into the prompt before each LLM call:
//
//	memories := agent.NewTaskMemoryManager(embedder, agent.NewInMemoryVectorStore(), agent.SimpleDuplicateChecker{})
//	clientWithMemory := agent.NewMutatingLLMClient(
//	    client,
//	    agent.NewMemoryInjector(memories, 3),
//	)
//
//	_, _, _ = memories, clientWithMemory, agent.New(clientWithMemory, defs, executor)
//
// Attach [WithWritePolicy] to [TaskMemoryManager] when only higher-value task
// completions should be stored as long-term memory. The built-in
// [ThresholdMemoryWritePolicy] is a good default when you want to skip trivial
// or low-detail task outcomes instead of saving every successful run.
//
// Retrieval-heavy applications can keep retrieval logic outside the agent while
// still sharing common contracts. [HybridRetriever] expresses a query-to-candidate
// retrieval step, [Reranker] refines those candidates, and [ChunkContextEnricher]
// can add document-aware context before indexing.
//
// # Retrieval terminology
//
// A few retrieval words show up often when building agent systems:
//
//   - A "chunk" is a small piece of source text that can be stored and retrieved
//     later. For example, one paragraph from a refund policy.
//   - "Chunk context enrichment" means attaching source-level context so the
//     chunk still makes sense by itself. For example, turning "refunds accepted
//     within 30 days" into "Refund Policy — refunds accepted within 30 days".
//   - "Lexical retrieval" means matching exact words or phrases.
//   - "Semantic retrieval" means matching by meaning, even when wording changes.
//   - "Hybrid retrieval" means combining more than one retrieval signal into a
//     single shortlist.
//   - "Reranking" means taking that rough shortlist and reordering it with a
//     slower, more precise second pass.
//   - "Dynamic tools" means showing the model only the tools that make sense in
//     the current phase of the workflow.
//   - "Grounding" means requiring the final answer to rely on authoritative
//     facts already captured in the run.
//   - A "circuit breaker" means stopping a repeated bad action instead of
//     letting the loop retry the same blocked path forever.
//
// `react-agent` does not force one retrieval stack. Instead it exposes small
// contracts so applications can plug in their own lexical search, vector search,
// reranking model, or indexing pipeline without changing the core agent loop.
//
// # Approval and compression terminology
//
// Two other concepts appear frequently in production agents:
//
//   - An "approval loop" pauses before a risky action, asks an external system
//     or human to decide, and then resumes or denies the action. In this
//     package that path runs through [ConfirmationCallback], [Suspend],
//     [InteractionRequest], [InteractionResponse], and [Agent.Resume].
//   - "Compression" or "context optimization" means shrinking noisy history
//     before the next model call so the useful parts stay visible. Common tools
//     here are [ContextOptimizer], [CompactionStrategy], and
//     [SummarizationStrategy].
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
