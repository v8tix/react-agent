# 🤖 react-agent

[![Go Reference](https://pkg.go.dev/badge/github.com/v8tix/react-agent.svg)](https://pkg.go.dev/github.com/v8tix/react-agent)
[![Go Report Card](https://goreportcard.com/badge/github.com/v8tix/react-agent)](https://goreportcard.com/report/github.com/v8tix/react-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Build AI agents that think before they act.**

Most LLM integrations are one-shot: send a prompt, get an answer. But complex questions require *reasoning* — looking things up, checking results, adjusting the plan. `react-agent` implements the **ReAct pattern** (Reason + Act), a technique where the model alternates between *thinking out loud* and *using tools* until it's confident enough to answer.

> 📄 *Based on ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) — Yao et al., 2022*

---

## 🧠 What is the ReAct pattern?

> Think of it like a **detective 🕵️** who never guesses. Instead of jumping to a conclusion, they follow a strict method: *form a hypothesis → gather evidence → revise → repeat* until the case is solved.

```mermaid
flowchart TD
    Q(["❓ User Question"])
    THINK["🧠 Think\nWhat do I need to find out?"]
    DECIDE{{"🤔 Need\na tool?"}}
    ACT["🔧 Act\nCall a tool"]
    OBSERVE["👁️ Observe\nRead tool output"]
    ANSWER["✅ Answer\nReturn final response"]
    LIMIT{{"🚧 Max steps\nreached?"}}
    ERR(["❌ ErrMaxStepsReached"])

    Q --> THINK
    THINK --> DECIDE
    DECIDE -->|"Yes 🛠️"| ACT
    ACT --> OBSERVE
    OBSERVE --> LIMIT
    LIMIT -->|"No"| THINK
    LIMIT -->|"Yes"| ERR
    DECIDE -->|"No, I know enough ✨"| ANSWER
```

---

## 🔍 A Concrete Example

> *"What was Apple's stock price the day the iPhone was announced?"*

A one-shot model will **guess**. A ReAct agent will **reason**:

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant A as 🤖 Agent
    participant L as 🧠 LLM
    participant T as 🔧 search_web

    U->>A: "What was Apple's stock price the day the iPhone was announced?"

    A->>L: Think 💭
    L-->>A: I need the announcement date first
    A->>T: search_web("first iPhone announcement date")
    T-->>A: "January 9, 2007" 📅

    A->>L: Think 💭 (now I have the date)
    L-->>A: Now I need the stock price on that date
    A->>T: search_web("AAPL stock price January 9 2007")
    T-->>A: "$11.74" 📈

    A->>L: Think 💭 (I have everything I need)
    L-->>A: ✅ Final answer

    A-->>U: "Apple's stock was $11.74 on Jan 9, 2007 — the day Steve Jobs unveiled the iPhone."
```

> 💡 Notice how the agent **builds on previous observations** — each step's result is fed back into the next Think. The LLM never loses context.

---

## 📦 Installation

```bash
go get github.com/v8tix/react-agent
```

---

## 🚀 Quick Start

Here's a complete example: a **research assistant** 🔬 that can search the web and do math to answer complex questions.

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/openai/openai-go"
    agent "github.com/v8tix/react-agent"
)

func main() {
    ctx := context.Background()

    // 1️⃣ Wrap your LLM (any OpenAI-compatible endpoint, including LiteLLM)
    openaiClient := openai.NewClient() // reads OPENAI_API_KEY from env
    client := agent.NewLiteLLMClient(openaiClient, "gpt-4o-mini")

    // 2️⃣ Declare the tools the agent can use (OpenAI JSON-schema format)
    defs := []agent.ToolDefinition{
        {
            Name:        "search_web",
            Description: "Search the web for up-to-date information",
            Parameters: map[string]any{
                "type": "object",
                "properties": map[string]any{
                    "query": map[string]any{"type": "string", "description": "The search query"},
                },
                "required": []string{"query"},
            },
        },
        {
            Name:        "calculator",
            Description: "Evaluate a simple arithmetic expression",
            Parameters: map[string]any{
                "type": "object",
                "properties": map[string]any{
                    "expression": map[string]any{"type": "string", "description": "e.g. '42 * 1.2'"},
                },
                "required": []string{"expression"},
            },
        },
    }

    // 3️⃣ Wire up the executor — your code that actually runs the tools
    executor := &myExecutor{}

    // 4️⃣ Build the agent with a fluent builder chain
    a := agent.New(client, defs, executor).
        WithInstructions("You are a precise research assistant. Always verify facts before answering.").
        WithMaxSteps(10)

    // 5️⃣ Run! Returns result, a replayable event stream, and any error.
    result, events, err := a.Run(ctx, "How many seconds did it take the Voyager 1 spacecraft to travel 1 AU?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("🏁 Answer:", result.Output)
    _ = events // see "Observability" section below
}
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Your Code"
        U(["👤 Caller"])
        EX["🔧 ToolExecutor\nimpl"]
        B["🪝 BeforeToolCallback"]
        AFT["🪝 AfterToolCallback"]
        HITL["✅ External approver / UI"]
    end

    subgraph "react-agent"
        AG["🤖 Agent\norchestrator"]
        EC["📋 ExecutionContext\nmessage history"]
        LP["🔁 ReAct Loop\nstep controller"]
        IR["⏸️ InteractionRequest /\nSuspendedRun"]
    end

    subgraph "External"
        LLM["🧠 LLM\n(OpenAI / LiteLLM)"]
        TOOLS["🛠️ Tools\n(search, DB, APIs...)"]
    end

    U -->|"New(...).WithX().Run()"| AG
    U -->|"Resume(...)"| AG
    AG --> EC
    AG --> LP
    SR["🧵 SessionRunner"]
    SM["💾 SessionManager"]
    MUT["🧠 MutatingLLMClient /\nRequestMutators"]
    MEM["📚 TaskMemoryManager /\nMemoryInjector"]
    OPT["🪶 ContextOptimizer /\nSummarization"]
    SR -->|"Run/Resume"| AG
    SR -->|"load + save"| SM
    LP -->|"prepare request"| MUT
    MUT -->|"inject memories"| MEM
    MUT -->|"shrink / summarize"| OPT
    MUT -->|"Generate(mutated request)"| LLM
    LLM -->|"ToolCall or Answer"| LP
    LP -->|"BeforeTool(call)"| B
    B -->|"override / continue"| LP
    B -.->|"Suspend(req)"| IR
    IR -.->|"emit + return"| U
    U -.->|"approve / deny"| HITL
    HITL -.->|"InteractionResponse"| AG
    LP -->|"Execute(calls)"| EX
    EX -->|"dispatch"| TOOLS
    TOOLS -->|"results"| EX
    EX -->|"ToolResult[]"| LP
    LP -->|"AfterTool(result)"| AFT
    AFT -->|"replace / keep"| LP
    LP -->|"*Result"| AG
    AG -->|"*Result, Observable, error"| U
```

---

## 🧾 Terminology Guide

These terms show up across the library because they describe the **shape of agent work**, not one specific app.

| Term | Plain-English meaning | Friendly example | Related API |
|------|------------------------|------------------|-------------|
| **Chunk** | a small piece of source text you can retrieve later | one paragraph from a refund policy | your app's indexing layer |
| **Chunk context enrichment** | adding source details so a chunk still makes sense by itself | `"Refund Policy — refunds accepted within 30 days"` instead of just `"refunds accepted within 30 days"` | `ChunkContextEnricher` |
| **Lexical retrieval** | finding results by exact words or phrases | query `"refund policy"` matches a chunk containing those exact words | app-defined retriever |
| **Semantic retrieval** | finding results by meaning, even if the wording changes | query `"money back rules"` still finds the refund chunk | embedder + vector store |
| **Hybrid retrieval** | combining more than one retrieval signal into one shortlist | exact words + meaning together | `HybridRetriever` |
| **Reranking** | taking a rough shortlist and reordering it with a stronger second pass | top 20 search hits become the best 3 | `Reranker` |
| **Approval loop** | pausing before a risky action so a human can approve it | ask before `delete_file` or `charge_card` | `ConfirmationCallback`, `Suspend`, `Resume` |
| **Callback** | a hook that can inspect, block, or rewrite work around tool execution | reject a tool call or shorten a huge tool result | `BeforeToolCallback`, `AfterToolCallback` |
| **Dynamic tools** | showing the model only the tools that make sense right now | planning turn sees `create_tasks`, recovery turn sees `research_conversion` | `WithDynamicToolsCallback` |
| **Workflow-owned control** | keeping business rules in your app while the library keeps running the loop | the workflow decides "plan -> convert -> fallback -> verify", not the generic runtime | callbacks + session/state in your app |
| **Deterministic phase** | a step where the workflow should decide what happens next, not the model | after an unsupported conversion, force fallback instead of asking the model to choose | dynamic tools + callbacks |
| **Grounding** | making the final answer explicitly rely on authoritative facts already gathered | reject an answer that ignores the recovered meter value | `FinalAnswerCallback` |
| **Circuit breaker** | stopping a repeated bad action instead of looping forever | block the same illegal tool twice, then fail loudly | `BeforeToolCallback` |
| **Compression / context optimization** | shrinking noisy history before the next LLM call | turn a 2-page HTML result into 3 useful lines | `ContextOptimizer`, `CompactionStrategy`, `SummarizationStrategy` |
| **Task memory** | storing solved-task patterns so the agent can reuse them later | "we solved a similar outage last week" | `TaskMemoryManager`, `MemoryInjector` |
| **Selective write** | storing only high-value memories instead of every completed task | skip saving `"what is 2+2"` but keep `"how we debugged the checkout outage"` | `MemoryWritePolicy`, `ThresholdMemoryWritePolicy` |

> 🙂 Friendly rule: **retrieve wide, then narrow; pause before risky actions; shrink context before it becomes noise.**

### Sequence: chunk context enrichment

Use this when a raw chunk is too small to stand on its own.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant E as 🧩 ChunkContextEnricher
    participant I as 🗂️ Index

    App->>E: EnrichChunk("refunds accepted within 30 days", metadata)
    E-->>App: "Refund Policy — refunds accepted within 30 days"
    App->>I: store enriched chunk
```

Without enrichment, the chunk may be technically correct but hard to understand once it is separated from the full document.

### Sequence: hybrid retrieval

Use this when exact wording matters **and** meaning matters.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant H as 🔀 HybridRetriever
    participant L as 🔎 Lexical search
    participant S as 🧠 Semantic search

    App->>H: Retrieve("money back rules", 10)
    H->>L: exact-word lookup
    L-->>H: candidates with phrase overlap
    H->>S: meaning-based lookup
    S-->>H: candidates with semantic similarity
    H-->>App: merged RetrievalCandidate list
```

Think of hybrid retrieval as **two flashlights pointed at the same shelf**: one catches exact labels, the other catches similar ideas.

### Sequence: reranking

Use this when the first retrieval pass is broad but not precise enough.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant R as 🎯 Reranker

    App->>R: Rerank(query, roughTopK, 3)
    R-->>App: better-ordered top 3
```

The common rhythm is: **retrieve 20 quickly, rerank to 3 carefully**.

### Sequence: approval loop

Use this when a tool can do something expensive, destructive, or externally visible.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant A as 🤖 Agent
    participant C as ✅ ConfirmationCallback
    participant UI as 🙋 Human approver
    participant X as 🔧 ToolExecutor

    App->>A: "Delete danger.txt"
    A->>C: BeforeTool(delete_file)
    C-->>A: Suspend(InteractionRequest)
    A-->>UI: approval needed
    UI-->>A: Resume(approved=true)
    A->>X: Execute(delete_file)
    X-->>A: ToolResult(success)
    A-->>App: final answer
```

If the answer is "no", the callback can return a synthetic error result instead of letting the tool run.

### Sequence: compression and context optimization

Use this when tool output or long conversations start crowding out the useful parts.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant M as 🪄 MutatingLLMClient
    participant O as 🪶 ContextOptimizer
    participant C as 📦 Compaction / summary strategy
    participant L as 🧠 LLM

    App->>M: Generate(request)
    M->>O: Mutate(request)
    O->>C: shrink bulky history
    C-->>O: trimmed request
    O-->>M: optimized request
    M->>L: Generate(optimized request)
```

This is less about deleting information and more about **keeping the signal while dropping the clutter**.

> 💡 `react-agent` intentionally keeps retrieval, reranking, approvals, and context control as small contracts. You can plug in your own search stack, vector DB, safety policy, or compression strategy without changing the core ReAct loop.

### Sequence: workflow-owned control

Use this when the loop is still useful, but some steps must follow a strict app-defined path.

```mermaid
flowchart TD
    U["👤 User asks for a bounded workflow"] --> P["🗂️ Planning phase\nshow only create_tasks"]
    P --> DC["📏 Direct conversion phase\ncontroller emits convert_units"]
    DC --> DECIDE{{"unsupported\nconversion?"}}
    DECIDE -->|"yes"| FC["🛟 Fallback phase\ncontroller emits research_conversion"]
    DECIDE -->|"no"| E["🔎 Evidence phase\nshow / emit gather_fact"]
    FC --> E
    E --> G["🧾 Final answer gate\ncheck grounding"]
    G -->|"grounded"| A["✅ Final answer"]
    G -->|"not grounded"| R["↩️ corrective user message"]
    R --> A
```

Friendly mental model: the library still runs the **same loop**, but your workflow can narrow the lane. The model keeps its reasoning ability, while your app decides which parts are too important to leave open-ended.

---

## 🔧 Implementing ToolExecutor

The only interface you **must** implement:

```go
type ToolExecutor interface {
    Execute(ctx context.Context, calls []model.ToolCall) ([]model.ToolResult, error)
}
```

A typical implementation dispatches by tool name:

```go
type myExecutor struct {
    searcher WebSearcher
    calc     Calculator
}

func (e *myExecutor) Execute(ctx context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
    results := make([]model.ToolResult, len(calls))
    for i, call := range calls {
        output, err := e.dispatch(ctx, call.Name, call.Arguments)
        if err != nil {
            results[i] = model.ToolResult{
                ID: call.ID, Name: call.Name,
                Status: "error", Content: []string{err.Error()},
            }
            continue
        }
        results[i] = model.ToolResult{
            ID: call.ID, Name: call.Name,
            Status: "success", Content: []string{output},
        }
    }
    return results, nil
}

func (e *myExecutor) dispatch(ctx context.Context, name, args string) (string, error) {
    switch name {
    case "search_web":
        return e.searcher.Search(ctx, args)
    case "calculator":
        return e.calc.Eval(args)
    default:
        return "", fmt.Errorf("unknown tool: %s", name)
    }
}
```

> 💡 `ToolExecutor` is **the integration seam** — plug in MCP, LangChain tools, a local SQLite, a REST API, anything. The agent doesn't care what's behind it.

---

## 🎓 Learning Flows

> 💡 **Study by scenario, not by type.** Pick the flow that looks like your app, copy the pattern, and then mix flows together as your agent grows up.

### ❓ Where should I start?

```mermaid
flowchart TD
    START{{"What are you building?"}}

    START -->|"One-shot assistant"| BASIC["🏃 Flow 1 + 🔧 Flow 2"]
    START -->|"Production chatbot"| CHAT["🧵 Flow 5 + 🪶 Flow 6"]
    START -->|"Agent automation"| AUTO["🔧 Flow 2 + 📚 Flow 7"]
    START -->|"Human-supervised agent"| SAFE["⏸️ Flow 4 + 🧵 Flow 5"]
    START -->|"Need observability"| OBS["📡 Flow 3 + 📊 Flow 8"]
    START -->|"Durable production agent"| DURABLE["🔄 Flow 9 + 🧵 Flow 5"]
    START -->|"Bounded workflow with rules"| CONTROL["🧭 Flow 10 + 🔧 Flow 2"]
```

| Flow | Best for | Complexity | Core pieces |
|------|----------|------------|-------------|
| **Flow 1 — 🏃 Basic Run** | direct answers, no tools | ⭐ | `Agent`, `LLMClient` |
| **Flow 2 — 🔧 Tool Use** | search, APIs, calculators | ⭐⭐ | `ToolExecutor`, `ToolDefinition` |
| **Flow 3 — 📡 Event Stream** | logs, tracing, dashboards | ⭐⭐ | `Observable`, `AgentEvent` |
| **Flow 4 — ⏸️ Approval Loop** | risky or destructive tools | ⭐⭐⭐ | `Suspend`, `Resume`, callbacks |
| **Flow 5 — 🧵 Session Continuity** | chat, multi-turn memory | ⭐⭐⭐ | `SessionRunner`, `SessionManager` |
| **Flow 6 — 🪶 Context Optimization** | long conversations | ⭐⭐⭐⭐ | `MutatingLLMClient`, strategies |
| **Flow 7 — 📚 Long-Term Memory** | learning from past tasks | ⭐⭐⭐⭐ | `TaskMemoryManager`, `MemoryInjector` |
| **Flow 8 — 📊 Request Logging** | debugging prompt pipelines | ⭐⭐⭐ | `WithLogger`, `WithMutatorLogger` |
| **Flow 9 — 🔄 Durable Workflows** | persistent sessions + selective memory | ⭐⭐⭐⭐ | `SessionRunner`, persister, memory policies |
| **Flow 10 — 🧭 Workflow-Owned Control** | bounded flows with deterministic steps | ⭐⭐⭐⭐ | dynamic tools, callbacks, final-answer gates |

---

### Flow 1 — 🏃 Basic Run: no tools, just reasoning

**Study this flow when:** you want the cleanest possible setup — one prompt in, one answer out.

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant A as 🤖 Agent
    participant L as 🧠 LLM

    U->>A: "What is the capital of France?"
    A->>L: Generate(request)
    L-->>A: "Paris."
    A-->>U: Result{Output: "Paris."}
```

```go
a := agent.New(client, nil, nil).
    WithInstructions("You are a helpful assistant.")

result, _, err := a.Run(ctx, "What is the capital of France?")
if err != nil {
    log.Fatal(err)
}

fmt.Println(result.Output)
fmt.Println("tool called:", result.ToolCalled)
```

> 🙂 Friendly rule of thumb: start here first. If this solves your use case, you probably do **not** need sessions, approvals, or memory yet.

---

### Flow 2 — 🔧 Tool Use: multi-step reasoning with evidence

**Study this flow when:** your model needs to look something up instead of guessing.

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant A as 🤖 Agent
    participant L as 🧠 LLM
    participant X as 🔧 ToolExecutor
    participant T as 🛠️ Tool

    U->>A: Ask a question that needs external data
    A->>L: Think
    L-->>A: ToolCall(search_web)
    A->>X: Execute(tool calls)
    X->>T: dispatch
    T-->>X: result
    X-->>A: ToolResult[]
    A->>L: Observe and think again
    L-->>A: Final answer
    A-->>U: Result
```

```go
defs := []model.ToolDefinition{{
    Name:        "search_web",
    Description: "Search the web for current information",
    Parameters: map[string]any{
        "type": "object",
        "properties": map[string]any{
            "query": map[string]any{"type": "string"},
        },
        "required": []string{"query"},
    },
}}

a := agent.New(client, defs, executor).
    WithInstructions("Verify facts before answering.")

result, _, err := a.Run(ctx, "What was Apple's stock price the day the iPhone was announced?")
if err != nil {
    log.Fatal(err)
}

fmt.Println(result.Output)
```

**What to study here:**

1. The agent writes `ToolCall` items into history before execution.
2. Your `ToolExecutor` decides how each tool name is actually dispatched.
3. Tool failures are still useful — the model can read them and recover on the next step.

---

### Flow 3 — 📡 Event Stream: observability without extra plumbing

**Study this flow when:** you want to see what the agent is doing while keeping the core orchestration code unchanged.

`Run()` returns three values:

```go
result, events, err := a.Run(ctx, question)
//        ^^^^^^
//        rxgo.Observable — a replayable stream of everything the agent did
```

```mermaid
timeline
    title Events emitted during one agent run
    RunStart    : 🚀 RunStartEvent
    Step 1      : 📍 StepStartEvent
                : 🧠 LLMCallEvent
                : 🪝 CallbackEvent
                : 🔧 ToolExecEvent
                : 📍 StepEndEvent
    Step 2      : 📍 StepStartEvent
                : 🧠 LLMCallEvent
                : ⏸️ InteractionRequestedEvent
                : ▶️ InteractionResumedEvent
                : 🔧 ToolExecEvent
                : 📍 StepEndEvent
    RunEnd      : 🏁 RunEndEvent
```

| Event | Why you care |
|------|---------------|
| `RunStartEvent` | identify a run and the original user question |
| `StepStartEvent` / `StepEndEvent` | measure loop progress |
| `LLMCallEvent` | track model latency and failures |
| `CallbackEvent` | observe approval or rewrite hooks |
| `ToolExecEvent` | track tool batches and latency |
| `InteractionRequestedEvent` | surface human input requests |
| `InteractionResumedEvent` | confirm resume handoff |
| `RunEndEvent` | capture final result or terminal error |

```go
result, events, err := a.Run(ctx, question)
if err != nil {
    log.Fatal(err)
}

for item := range events.Observe() {
    switch e := item.V.(type) {
    case agent.RunStartEvent:
        slog.Info("run started", "run_id", e.RunID, "question", e.UserMessage)
    case agent.ToolExecEvent:
        slog.Info("tool exec", "tools", e.ToolNames, "latency_ms", e.Latency.Milliseconds())
    case agent.RunEndEvent:
        slog.Info("run ended", "err", e.Err)
    }
}

fmt.Println(result.Output)
```

> 🧊 **Cold & replayable** — every `Observe()` call replays the full run from the beginning, so one subscriber can log while another records metrics.

---

### Flow 4 — ⏸️ Approval Loop: pause, ask a human, continue safely

**Study this flow when:** a tool can delete, charge, publish, or otherwise do something you want a human to approve.

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant A as 🤖 Agent
    participant C as 🪝 BeforeToolCallback
    participant UI as ✅ External approver
    participant X as 🔧 ToolExecutor

    U->>A: "Delete danger.txt"
    A->>C: BeforeTool(delete_file)
    C-->>A: Suspend(InteractionRequest)
    A-->>UI: InteractionRequestedError
    UI-->>A: Resume(approved=true/false)
    A->>C: BeforeTool(delete_file) again
    alt approved
        C-->>A: continue
        A->>X: Execute(delete_file)
        X-->>A: ToolResult(success)
    else denied
        C-->>A: ToolResult(error)
    end
    A-->>U: final answer
```

```go
approval := agent.NewConfirmationCallback(agent.StaticApprovalPolicy{
    "delete_file": {
        MessageTemplate: "Approve deleting this file?",
        DeniedMessage:   "Deletion cancelled by user.",
    },
})

a := agent.New(client, defs, executor).
    WithInstructions("Ask for approval before destructive tools.").
    WithBeforeToolCallbacks(approval).
    WithMaxSteps(8)
```

```go
result, _, err := a.Run(ctx, "Delete danger.txt")
if err != nil {
    var suspended *agent.InteractionRequestedError
    if errors.As(err, &suspended) {
        approved := true
        result, _, err = a.Resume(ctx, suspended.Suspended, agent.InteractionResponse{
            RequestID: suspended.Suspended.Interaction.ID,
            Approved:  &approved,
        })
    }
}
```

**Nice built-in detail:** `ConfirmationCallback` automatically redacts sensitive-looking tool arguments before they land in `InteractionRequest.Payload`.

| Payload pattern | How it is handled |
|----------------|-------------------|
| `api_key`, `token`, `password`, `secret` | replaced with `[redacted]` |
| path-like fields | sanitized before being surfaced |
| normal strings | truncated and cleaned for safe display |

---

### Flow 5 — 🧵 Session Continuity: multi-turn conversations without losing context

**Study this flow when:** your caller is not a single CLI invocation — it's a chat UI, API, or worker that talks to the same user over time.

```mermaid
sequenceDiagram
    participant App as 👤 App / API
    participant SR as 🧵 SessionRunner
    participant SM as 💾 SessionManager
    participant AG as 🤖 Agent

    App->>SR: Run(sessionID, userID, "My name is Alice")
    SR->>SM: GetOrCreate(sessionID, userID)
    SM-->>SR: session(events=[], state={})
    SR->>AG: replay events + run agent
    AG-->>SR: RunResult{Status: complete}
    SR->>SM: Save(updated events)
    SR-->>App: "Nice to meet you, Alice!"

    App->>SR: Run(sessionID, userID, "What's my name?")
    SR->>SM: Get(sessionID)
    SM-->>SR: session(previous events)
    SR->>AG: replay events + run agent
    AG-->>SR: RunResult{Status: complete, Output: "Your name is Alice."}
    SR->>SM: Save(updated events)
    SR-->>App: "Your name is Alice."
```

```go
sessions := agent.NewInMemorySessionManager()

runner := agent.NewSessionRunner(
    agent.New(client, defs, executor).WithMaxSteps(8),
    sessions,
    8,
)

first, err := runner.Run(ctx, "chat-42", "user-7", "My name is Alice")
if err != nil {
    log.Fatal(err)
}

second, err := runner.Run(ctx, "chat-42", "user-7", "What's my name?")
if err != nil {
    log.Fatal(err)
}

fmt.Println(first.Status)
fmt.Println(second.Output)
```

**Study notes:**

1. `SessionRunner` replays stored `model.Event` values into a fresh execution context.
2. The session store is pluggable — `InMemorySessionManager` is just the starter version.
3. If a session suspends, you get `StatusPending` and can continue later with `runner.Resume(...)`.

---

### Flow 6 — 🪶 Context Optimization: managing token budgets without going blind

**Study this flow when:** your conversation gets long, tool outputs get noisy, or the model starts forgetting important recent context.

```mermaid
flowchart LR
    MSG["📝 Current request"] --> MUT["MutatingLLMClient"]
    MUT --> INJ["MemoryInjector (optional)"]
    INJ --> OPT["ContextOptimizer"]
    OPT --> SW["SlidingWindowStrategy"]
    OPT --> CP["CompactionStrategy"]
    OPT --> SUM["SummarizationStrategy"]
    SUM --> LLM["🧠 Underlying LLMClient"]
```

```go
counter, err := agent.NewRequestTokenCounter("gpt-4o-mini")
if err != nil {
    log.Fatal(err)
}

optimizedClient := agent.NewMutatingLLMClient(
    client,
    agent.WithMutatorLogger(
        agent.NewContextOptimizer(
            counter,
            8_000,
            agent.NewSlidingWindowStrategy(8),
            agent.NewCompactionStrategy(),
        ),
        slog.Default(),
    ),
)

a := agent.New(optimizedClient, defs, executor).WithMaxSteps(12)
```

| Strategy | What it does | Best when |
|----------|---------------|-----------|
| `SlidingWindowStrategy` | keeps the last user turn plus a recent tail | you only need fresh context |
| `CompactionStrategy` | shrinks bulky tool payloads into short summaries | tools return huge blobs |
| `SummarizationStrategy` | moves older history into a generated summary | conversations are long but earlier turns still matter |

> 🧠 Think of this as luggage management for prompts: keep the passport, fold the clothes, and summarize the travel diary.

---

### Flow 7 — 📚 Long-Term Task Memory: learning from similar problems

**Study this flow when:** your agent solves recurring task shapes and you want it to reuse successful approaches next time.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant MM as 📚 TaskMemoryManager
    participant E as 🧠 Embedder
    participant VS as 🗂️ VectorStore
    participant MI as 🪄 MemoryInjector

    App->>MM: Save(TaskMemory)
    MM->>E: Embed("Task: ...")
    E-->>MM: vector
    MM->>VS: Search(vector, 3) for duplicates
    VS-->>MM: nearest memories
    MM->>VS: Add(VectorDocument)

    App->>MI: Mutate(request)
    MI->>MM: Search(last user message, topK)
    MM->>E: Embed(query)
    E-->>MM: vector
    MM->>VS: Search(vector, topK)
    VS-->>MM: relevant memories
    MM-->>MI: TaskMemory[]
    MI-->>App: instructions + <PAST_EXPERIENCES>
```

```go
memories := agent.NewTaskMemoryManager(
    embedder,
    agent.NewInMemoryVectorStore(),
    agent.SimpleDuplicateChecker{},
)

if _, saved, err := memories.Save(ctx, agent.TaskMemory{
    TaskSummary: "find capitals",
    Approach:    "used search",
    FinalAnswer: "Paris",
    IsCorrect:   true,
}); err != nil {
    log.Fatal(err)
} else if saved {
    fmt.Println("memory stored")
}

clientWithMemory := agent.NewMutatingLLMClient(
    client,
    agent.NewMemoryInjector(memories, 3),
)
```

| Field | What to put there |
|------|--------------------|
| `TaskSummary` | short description of the problem |
| `Approach` | how the agent solved it |
| `FinalAnswer` | the final answer or resolution |
| `IsCorrect` | whether the solution should be reused confidently |
| `ErrorAnalysis` | what went wrong when a result was bad |

Stored memories and injected text are sanitized before reuse, so prompt-injection strings and sensitive-looking paths do not get copied back into the model context verbatim.

---

### Sequence: selective memory write

Use this when some task outcomes are worth keeping long-term, but many are just noise.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant MM as 📚 TaskMemoryManager
    participant P as 🎯 MemoryWritePolicy
    participant VS as 🗂️ VectorStore

    App->>MM: Save(TaskMemory)
    MM->>P: Decide(memory)
    alt high value
        P-->>MM: ShouldStore=true
        MM->>VS: Add(memory)
        VS-->>MM: stored
        MM-->>App: saved=true
    else low value
        P-->>MM: ShouldStore=false
        MM-->>App: saved=false
    end
```

Think of this as **memory budgeting**: keep the hard-won lessons, skip the trivia.

```go
memories := agent.NewTaskMemoryManager(
    embedder,
    agent.NewInMemoryVectorStore(),
    agent.SimpleDuplicateChecker{},
).WithWritePolicy(agent.NewThresholdMemoryWritePolicy(0.6))

_, saved, err := memories.Save(ctx, agent.TaskMemory{
    TaskSummary: "debug checkout timeout with retries",
    Approach:    "trace logs, isolate retry loop, then patch",
    FinalAnswer: "fixed with bounded retry backoff",
    IsCorrect:   true,
})
if err != nil {
    log.Fatal(err)
}
fmt.Println("saved:", saved)
```

---

### Flow 8 — 📊 Request Logging: see the mutator pipeline in action

**Study this flow when:** you want to debug why a prompt got shorter, why memories were injected, or why an approval flow suspended.

```mermaid
sequenceDiagram
    participant App as 👤 App
    participant M1 as 🪄 WithMutatorLogger
    participant MI as 📚 MemoryInjector
    participant M2 as 🪄 WithMutatorLogger
    participant CO as 🪶 ContextOptimizer
    participant L as 🧠 LLM

    App->>M1: Mutate(request)
    M1->>MI: start
    MI-->>M1: finish
    App->>M2: Mutate(request)
    M2->>CO: start
    CO-->>M2: finish
    App->>L: Generate(mutated request)
```

```go
logger := slog.Default()

memories := agent.NewTaskMemoryManager(embedder, agent.NewInMemoryVectorStore(), agent.SimpleDuplicateChecker{}).
    WithLogger(logger)

counter, err := agent.NewRequestTokenCounter("gpt-4o-mini")
if err != nil {
    log.Fatal(err)
}

clientWithPipelineLogs := agent.NewMutatingLLMClient(
    client,
    agent.WithMutatorLogger(agent.NewMemoryInjector(memories, 3), logger),
    agent.WithMutatorLogger(
        agent.NewContextOptimizer(
            counter,
            8_000,
            agent.NewSlidingWindowStrategy(8),
            agent.NewCompactionStrategy(),
        ).WithLogger(logger),
        logger,
    ),
)
```

**Typical things you'll see in logs:**

1. `memory_search_start` / `memory_search_end`
2. `mutator_start` / `mutator_finish`
3. `context_optimize_start`
4. `context_strategy_apply` / `context_strategy_applied`
5. `session_run_start` / `session_run_end`
6. `approval_requested`

---

### Flow 9 — 🔄 Durable Workflows: persistent sessions, selective memory, and cache-friendly prompts

**Study this flow when:** your agent must survive restarts, remember only useful lessons, and keep repeated prompt setup cheap.

```mermaid
sequenceDiagram
    participant App as 👤 App / API
    participant SR as 🧵 SessionRunner
    participant PS as 💾 SessionPersister
    participant MC as 🪄 MutatingLLMClient
    participant SPD as 📍 StablePrefixDetector
    participant AG as 🤖 Agent
    participant MM as 📚 TaskMemoryManager
    participant WP as 🎯 MemoryWritePolicy

    App->>SR: Run(sessionID, userID, question)
    SR->>PS: LoadSession(sessionID)
    PS-->>SR: prior events + state
    SR->>AG: replay and continue workflow
    AG->>MC: Generate(request)
    MC->>SPD: Detect stable prefix
    SPD-->>MC: reusable prefix
    MC-->>AG: optimized request
    AG-->>SR: result
    SR->>MM: Save(TaskMemory)
    MM->>WP: Decide(memory value)
    WP-->>MM: store or skip
    SR->>PS: SaveSession(updated)
    SR-->>App: result
```

This flow combines four ideas that often show up together in production:

1. **Persistent sessions** keep the conversation alive across restarts or separate workers.
2. **Selective memory writes** stop long-term memory from filling with low-value tasks.
3. **Stable prefix detection** helps your LLM client identify the reusable part of a request for prompt-caching workflows.
4. **Workflow composition** lets later turns depend on facts or state captured earlier in the same session.

#### Persistent sessions

`InMemorySessionManager` is great for local development. In production, swap in `NewPersistedSessionManager` so a restart does not erase the conversation.

```go
type myPersister struct{}

func (p *myPersister) SaveSession(ctx context.Context, session agent.Session) error {
    return saveToStore(ctx, session) // your DB, Redis, S3, etc.
}

func (p *myPersister) LoadSession(ctx context.Context, sessionID string) (agent.Session, error) {
    return loadFromStore(ctx, sessionID)
}

sessions := agent.NewPersistedSessionManager(&myPersister{})
runner := agent.NewSessionRunner(a, sessions, 8)
```

#### Stable prefixes and caching

A **stable prefix** is the part of the request that barely changes across turns: instructions, fixed tool setup, and maybe the early session scaffold. If your provider supports prompt caching, this is the part you usually want to mark as reusable.

Friendly example:

- stable: `"You are a support assistant..."` + fixed tool definitions
- unstable: the newest user message, fresh tool output, current turn state

`react-agent` does not force one provider-specific caching implementation. It gives you the seam so your own LLM client can use the detected prefix.

#### Workflow composition

Treat `Session.State` like shared scratch space across turns.

```go
first, _ := runner.Run(ctx, "order-42", "user-9", "Find the root cause")
session, _ := sessions.Get("order-42")
session.State["root_cause"] = first.Output
_ = sessions.Save(session)

second, _ := runner.Run(ctx, "order-42", "user-9", "Propose the safest fix")
_ = second
```

That pattern is what turns a chat session into a **multi-step workflow**.

---

### Flow 10 — 🧭 Workflow-Owned Control: deterministic phases inside an open-ended loop

**Study this flow when:** your agent still benefits from ReAct, but some steps are too important, expensive, or risky to leave fully open-ended.

Good examples:

1. plan first, then force a conversion step
2. switch to a fallback path after a permanent tool failure
3. require supporting facts before the answer
4. reject a final answer that ignores authoritative values already collected

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant A as 🤖 Agent
    participant D as 🧰 Dynamic tool callback
    participant B as 🪝 Before/After callbacks
    participant X as 🔧 ToolExecutor
    participant F as ✅ Final answer gate

    U->>A: "Run the bounded workflow"
    A->>D: Which tools are visible now?
    D-->>A: create_tasks only
    A->>X: Execute(create_tasks)
    B-->>A: move to direct conversion phase
    A->>D: Which tools are visible now?
    D-->>A: convert_units only
    A->>X: Execute(convert_units)
    B-->>A: unsupported conversion -> fallback phase
    A->>D: Which tools are visible now?
    D-->>A: research_conversion only
    A->>X: Execute(research_conversion)
    B-->>A: collect recovered facts
    A->>F: Proposed final answer
    F-->>A: accept or reject with corrective message
    A-->>U: grounded final answer
```

This is the pattern we used for a bounded adaptive workflow:

- the **library** still owns the loop, history, callbacks, suspension, and resume
- the **workflow** owns phases, allowed tools, fallback rules, circuit breakers, and grounding checks

```go
phaseTracker := newMyWorkflowStateMachine()

a := agent.New(client, defs, executor).
    WithDynamicToolsCallback(func(execCtx *agent.ExecutionContext) []model.ToolDefinition {
        return phaseTracker.AllowedTools(defs)
    }).
    WithBeforeToolCallbacks(phaseTracker).
    WithAfterToolCallbacks(phaseTracker).
    WithFinalAnswerCallbacks(myWorkflowGate{tracker: phaseTracker})
```

**What to study here:**

1. Use **dynamic tools** when the model should only see the tools that make sense in the current phase.
2. Use **before/after callbacks** when the workflow needs to track failures, state transitions, or repeated bad behavior.
3. Use a **final answer callback** when the answer must mention or rely on specific verified facts.
4. Keep the workflow rules in your app code — `react-agent` gives you the seams, not one hardcoded business process.

> 🙂 Friendly rule: let the model stay flexible where reasoning helps, but take the wheel for the steps that must be correct in the same way every time.

---

## 🔗 Flow Combinations

Real systems usually mix more than one flow:

| Your use case | Combine these flows | Why |
|---------------|---------------------|-----|
| **Production chatbot** | Flow 3 + Flow 5 + Flow 6 + Flow 9 | observe runs, persist turns, keep prompts small, and save only useful lessons |
| **Risky automation** | Flow 2 + Flow 4 + Flow 8 | use tools, gate them, log every decision |
| **Research assistant** | Flow 2 + Flow 3 + Flow 7 | fetch evidence, inspect reasoning, reuse good prior work |
| **Support agent** | Flow 5 + Flow 6 + Flow 7 + Flow 9 | remember the conversation, manage token budget, learn from resolved cases, and survive restarts |
| **Bounded adaptive workflow** | Flow 2 + Flow 3 + Flow 10 | let the model reason, but keep critical phases deterministic and grounded |

> 🧩 The library is intentionally composable: sessions, approvals, mutators, memory, and event streams are designed to stack cleanly instead of forcing one giant framework.

---

## 🕹️ Manual Step Control

`Step` is still available when you want to drive the loop yourself — useful for streaming UI updates, custom checkpoints, or research/debug flows where you want to inspect each step manually:

```go
execCtx := agent.NewExecutionContextForTest()
execCtx.AddEvent("user", model.Message{Role: "user", Content: "Plan a 3-day trip to Kyoto"})

for execCtx.CurrentStep() < 15 {
    if err := a.Step(ctx, execCtx); err != nil {
        break
    }
    if execCtx.Done() {
        break
    }

    latest := execCtx.Events()[len(execCtx.Events())-1]
    fmt.Printf("step %d: %s emitted %d item(s)\n", execCtx.CurrentStep(), latest.Author, len(latest.Content))

    execCtx.IncrementStep()
}
```

---

## 🔍 Inspecting the Reasoning Trail

Every run keeps a full, ordered history of messages, tool calls, and tool results in `result.Context`:

```go
for _, event := range result.Context.Events() {
        fmt.Printf("[%s] at %s\n", event.Author, event.Timestamp.Format(time.RFC3339))
    for _, item := range event.Content {
        switch v := item.(type) {
        case model.Message:
            fmt.Printf("  💬 message: %s\n", v.Content)
        case model.ToolCall:
            fmt.Printf("  🔧 tool_call: %s(%s)\n", v.Name, v.Arguments)
        case model.ToolResult:
            fmt.Printf("  📊 tool_result: [%s] %v\n", v.Status, v.Content)
        }
    }
}
```

> 💡 Use this for **debugging**, **audit logs**, or displaying the agent's "chain of thought" to end users.

---

## 🏛️ Design Reference

```mermaid
classDiagram
    class Agent {
        +Run(ctx, question) Result, Observable, error
        +Resume(ctx, suspended, response) Result, Observable, error
        +Step(ctx, execCtx) error
        +Think(ctx, execCtx) error
        +Act(ctx, execCtx) error
        +WithInstructions(string) Agent
        +WithMaxSteps(int) Agent
        +WithBeforeToolCallbacks(...BeforeToolCallback) Agent
        +WithAfterToolCallbacks(...AfterToolCallback) Agent
    }

    class LLMClient {
        <<interface>>
        +Generate(ctx, req) Response, error
    }

    class ToolExecutor {
        <<interface>>
        +Execute(ctx, calls) ToolResults, error
    }

    class BeforeToolCallback {
        <<interface>>
        +BeforeTool(ctx, execCtx, call) ToolResult, error
    }

    class AfterToolCallback {
        <<interface>>
        +AfterTool(ctx, execCtx, result) ToolResult, error
    }

    class ExecutionContext {
        +Events() []Event
        +AddEvent(author, items)
        +CurrentStep() int
        +IncrementStep()
        +PendingInteraction() *InteractionRequest, bool
        +InteractionResponse(requestID) InteractionResponse, bool
    }

    class SuspendedRun {
        +Context ExecutionContext
        +Interaction InteractionRequest
    }

    class InteractionRequest {
        +ID string
        +Kind string
        +Prompt string
        +ToolCallID string
        +ToolName string
    }

    class InteractionResponse {
        +RequestID string
        +Approved *bool
        +Value string
    }

    class AgentEvent {
        <<interface sealed>>
        RunStartEvent
        StepStartEvent
        LLMCallEvent
        CallbackEvent
        ToolExecEvent
        InteractionRequestedEvent
        InteractionResumedEvent
        StepEndEvent
        RunEndEvent
    }

    Agent --> LLMClient : uses
    Agent --> ToolExecutor : uses
    Agent --> BeforeToolCallback : uses
    Agent --> AfterToolCallback : uses
    Agent --> ExecutionContext : owns
    Agent --> SuspendedRun : resumes
    Agent ..> AgentEvent : emits via Observable
    InteractionRequestedEvent --> InteractionRequest : carries
    InteractionResumedEvent --> InteractionResponse : carries
```

| Type | Role |
|------|------|
| `Agent` | 🤖 Orchestrator — owns the loop |
| `ExecutionContext` | 📋 Mutable run state — thread-safe event log |
| `Event` | 📝 Timestamped history entry (author + content items) |
| `LLMClient` | 🧠 Interface — swap any provider |
| `ToolExecutor` | 🔧 Interface — bring your own dispatch strategy |
| `LiteLLMClient` | 🔌 Concrete adapter for openai-go / LiteLLM proxy |
| `MutatingLLMClient` | 🧠 Request pipeline — inject memory, optimize, sanitize |
| `SessionRunner` | 🧵 Conversation wrapper — replay, persist, resume |
| `TaskMemoryManager` | 📚 Long-term memory store — save and search past tasks |
| `ConfirmationCallback` | ✅ Human approval gate for selected tools |
| `WithDynamicToolsCallback` | 🧰 Per-turn tool visibility — show only what this phase should allow |
| `AgentEvent` | 📡 Sealed sum type — emitted on the observable stream |

---

## License

MIT
