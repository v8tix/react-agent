package mcpadapter

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/reactivex/rxgo/v2"
	"github.com/v8tix/mcp-toolkit/handler"
	llmmodel "github.com/v8tix/mcp-toolkit/model"
	"github.com/v8tix/mcp-toolkit/observable"
	"github.com/v8tix/mcp-toolkit/registry"

	"github.com/v8tix/react-agent/model"
)

// indexedResult pairs a call-order index with its resolved ToolResult so that
// insertion order can be restored after rxgo.Merge delivers items in completion
// order (i.e. fastest tools first).
type indexedResult struct {
	idx    int
	result model.ToolResult
}

// RegistryExecutor implements model.ToolExecutor using mcp-toolkit's concurrent
// observable dispatch with retry and exponential backoff.
//
// Use NewRegistryExecutor to construct one, or call FromRegistry for a
// one-liner that also returns the tool definitions.
type RegistryExecutor struct {
	reg *registry.Registry
}

// NewRegistryExecutor creates a RegistryExecutor backed by reg.
func NewRegistryExecutor(reg *registry.Registry) *RegistryExecutor {
	return &RegistryExecutor{reg: reg}
}

// Defs converts a slice of mcp-toolkit ToolDefinitions to model.ToolDefinitions.
// Pass the result as the defs argument to agent.New().
//
//	defs := mcpadapter.Defs(reg.All())
func Defs(defs []llmmodel.ToolDefinition) []model.ToolDefinition {
	result := make([]model.ToolDefinition, len(defs))
	for i, d := range defs {
		result[i] = model.ToolDefinition{
			Name:        d.Function.Name,
			Description: d.Function.Description,
			Parameters:  d.Function.Parameters.ToMap(),
			Strict:      d.Function.Strict,
		}
	}
	return result
}

// FromRegistry is a convenience helper that returns both tool definitions and a
// RegistryExecutor from a single registry — the two values needed by agent.New().
//
//	defs, executor := mcpadapter.FromRegistry(reg)
//	a := agent.New(client, defs, executor, agent.WithInstructions("..."))
func FromRegistry(reg *registry.Registry) ([]model.ToolDefinition, model.ToolExecutor) {
	return Defs(reg.All()), NewRegistryExecutor(reg)
}

// Execute fans out all tool calls concurrently via rxgo.Merge, waits for all
// results, restores the original call order, and returns []model.ToolResult.
//
// Errors are never returned as Go errors — they are encoded into
// ToolResult.Status="error" so the agent loop can always continue the
// conversation and let the LLM reason about the failure.
func (e *RegistryExecutor) Execute(ctx context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
	if len(calls) == 0 {
		return nil, nil
	}

	observables := make([]rxgo.Observable, len(calls))
	for i, call := range calls {
		observables[i] = e.callObservable(ctx, i, call)
	}

	indexed := make([]indexedResult, 0, len(calls))
	for item := range rxgo.Merge(observables).Observe() {
		if item.E != nil {
			// Should not happen — callObservable encodes all errors into the
			// result. Guard here for safety.
			continue
		}
		if r, ok := item.V.(indexedResult); ok {
			indexed = append(indexed, r)
		}
	}

	sort.Slice(indexed, func(i, j int) bool { return indexed[i].idx < indexed[j].idx })

	results := make([]model.ToolResult, len(indexed))
	for i, r := range indexed {
		results[i] = r.result
	}
	return results, nil
}

// callObservable builds a cold rxgo.Observable for a single tool call.
// It emits exactly one indexedResult — never an error — so rxgo.Merge can
// always collect a result for every call regardless of failure.
func (e *RegistryExecutor) callObservable(ctx context.Context, idx int, call model.ToolCall) rxgo.Observable {
	errorResult := func(msg string) rxgo.Observable {
		return rxgo.Just(indexedResult{idx, model.ToolResult{
			ID:      call.ID,
			Name:    call.Name,
			Status:  "error",
			Content: []string{msg},
		}})()
	}

	tool, ok := e.reg.ByName(call.Name)
	if !ok {
		return errorResult(fmt.Sprintf("tool %q not found in registry", call.Name))
	}

	exec, ok := tool.(handler.ExecutableTool)
	if !ok {
		return errorResult(fmt.Sprintf("tool %q is not executable", call.Name))
	}

	rawArgs := call.Arguments

	// observable.Tool gets the retry-aware reactive path; plain ExecutableTool
	// gets a simple rxgo.Defer wrapper (one attempt, no backoff).
	var source rxgo.Observable
	if obsTool, ok := tool.(observable.Tool); ok {
		source = obsTool.ExecuteRx(ctx, rawArgs)
	} else {
		source = rxgo.Defer([]rxgo.Producer{
			func(_ context.Context, next chan<- rxgo.Item) {
				result, err := exec.Execute(ctx, rawArgs)
				if err != nil {
					next <- rxgo.Error(err)
					return
				}
				next <- rxgo.Of(result)
			},
		})
	}

	// Map the successful result to an indexedResult, serializing the tool output
	// to JSON so the LLM receives a consistent string representation.
	return source.Map(func(_ context.Context, item any) (any, error) {
		content, err := json.Marshal(item)
		if err != nil {
			return indexedResult{idx, model.ToolResult{
				ID:      call.ID,
				Name:    call.Name,
				Status:  "error",
				Content: []string{fmt.Sprintf("result serialisation error: %s", err.Error())},
			}}, nil
		}
		return indexedResult{idx, model.ToolResult{
			ID:      call.ID,
			Name:    call.Name,
			Status:  "success",
			Content: []string{string(content)},
		}}, nil
	}, rxgo.WithErrorStrategy(rxgo.ContinueOnError)).
		OnErrorReturn(func(err error) any {
			return indexedResult{idx, model.ToolResult{
				ID:      call.ID,
				Name:    call.Name,
				Status:  "error",
				Content: []string{fmt.Sprintf("execution error: %s", err.Error())},
			}}
		})
}
