package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/tiktoken-go/tokenizer"
	"github.com/v8tix/react-agent/model"
)

// TokenCounter estimates the prompt size of a request before it is sent to the LLM.
type TokenCounter interface {
	Count(req model.Request) (int, error)
}

// RequestMutator rewrites a request immediately before the delegated LLM call.
type RequestMutator interface {
	Mutate(ctx context.Context, req *model.Request) error
}

// OptimizationStrategy rewrites a request to reduce context size or noise.
type OptimizationStrategy interface {
	Optimize(ctx context.Context, req *model.Request) error
}

// SummaryGenerator compresses older events into a summary string.
type SummaryGenerator interface {
	Summarize(ctx context.Context, events []model.Event) (string, error)
}

// MutatingLLMClient applies request mutators and then forwards the request to
// another [LLMClient].
type MutatingLLMClient struct {
	delegate LLMClient
	mutators []RequestMutator
}

// NewMutatingLLMClient wraps an [LLMClient] with one or more request mutators.
func NewMutatingLLMClient(delegate LLMClient, mutators ...RequestMutator) *MutatingLLMClient {
	return &MutatingLLMClient{delegate: delegate, mutators: slices.Clone(mutators)}
}

// Generate clones the request, applies every mutator in order, and delegates the call.
func (c *MutatingLLMClient) Generate(ctx context.Context, req model.Request) (model.Response, error) {
	cloned := cloneRequest(req)
	for _, mutator := range c.mutators {
		if err := mutator.Mutate(ctx, &cloned); err != nil {
			return model.Response{}, err
		}
	}
	return c.delegate.Generate(ctx, cloned)
}

// RequestTokenCounter uses tiktoken-compatible tokenization to count request size.
type RequestTokenCounter struct{ codec tokenizer.Codec }

// NewRequestTokenCounter constructs a token counter for the given model name.
func NewRequestTokenCounter(modelName string) (*RequestTokenCounter, error) {
	codec, err := tokenizer.ForModel(tokenizer.Model(modelName))
	if err != nil {
		codec, err = tokenizer.Get(tokenizer.O200kBase)
		if err != nil {
			return nil, fmt.Errorf("get tokenizer codec: %w", err)
		}
	}
	return &RequestTokenCounter{codec: codec}, nil
}

// Count returns the approximate token count for instructions, events, and tools.
func (c *RequestTokenCounter) Count(req model.Request) (int, error) {
	total := 0
	add := func(text string) error {
		if text == "" {
			return nil
		}
		n, err := c.codec.Count(text)
		if err != nil {
			return err
		}
		total += n
		return nil
	}
	if err := add(req.Instructions); err != nil {
		return 0, err
	}
	for _, event := range req.Events {
		if err := add(event.Author); err != nil {
			return 0, err
		}
		for _, item := range event.Content {
			switch value := item.(type) {
			case model.Message:
				if err := add(value.Role + ":" + value.Content); err != nil {
					return 0, err
				}
			case model.ToolCall:
				if err := add(value.Name); err != nil {
					return 0, err
				}
				if err := add(string(value.Arguments)); err != nil {
					return 0, err
				}
			case model.ToolResult:
				if err := add(value.Name + ":" + value.Status); err != nil {
					return 0, err
				}
				if err := add(strings.Join(value.Content, "\n")); err != nil {
					return 0, err
				}
			}
		}
	}
	for _, tool := range req.Tools {
		payload, err := json.Marshal(tool)
		if err != nil {
			return 0, err
		}
		if err := add(string(payload)); err != nil {
			return 0, err
		}
	}
	return total, nil
}

// SlidingWindowStrategy keeps the latest user message plus a bounded tail of recent events.
type SlidingWindowStrategy struct{ windowSize int }

// NewSlidingWindowStrategy creates a sliding-window optimizer.
func NewSlidingWindowStrategy(windowSize int) SlidingWindowStrategy {
	return SlidingWindowStrategy{windowSize: windowSize}
}

// Optimize trims the request history while preserving the most recent user turn.
func (s SlidingWindowStrategy) Optimize(_ context.Context, req *model.Request) error {
	if s.windowSize <= 0 || len(req.Events) == 0 {
		return nil
	}
	userIdx := -1
	for i := len(req.Events) - 1; i >= 0; i-- {
		event := req.Events[i]
		if event.Author != "user" {
			continue
		}
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" {
				userIdx = i
				break
			}
		}
		if userIdx >= 0 {
			break
		}
	}
	if userIdx < 0 {
		return nil
	}
	preserved := []model.Event{req.Events[userIdx]}
	remaining := append([]model.Event(nil), req.Events[userIdx+1:]...)
	if len(remaining) > s.windowSize {
		remaining = remaining[len(remaining)-s.windowSize:]
	}
	req.Events = append(preserved, remaining...)
	return nil
}

// CompactionStrategy replaces bulky tool payloads with short, sanitized summaries.
type CompactionStrategy struct{}

// NewCompactionStrategy creates a compaction optimizer for large tool payloads.
func NewCompactionStrategy() CompactionStrategy { return CompactionStrategy{} }

// Optimize compacts selected tool calls and tool results in-place.
func (CompactionStrategy) Optimize(_ context.Context, req *model.Request) error {
	callArgs := map[string]map[string]any{}
	for eventIdx, event := range req.Events {
		for contentIdx, item := range event.Content {
			switch value := item.(type) {
			case model.ToolCall:
				args, err := decodeArguments(value.Arguments)
				if err != nil {
					return fmt.Errorf("decode tool call %s arguments: %w", value.ID, err)
				}
				callArgs[value.ID] = args
				if value.Name == "create_file" {
					args["content"] = "[Content saved to file]"
					value.Arguments, err = json.Marshal(args)
					if err != nil {
						return fmt.Errorf("encode compacted tool call %s arguments: %w", value.ID, err)
					}
					req.Events[eventIdx].Content[contentIdx] = value
				}
			case model.ToolResult:
				args := callArgs[value.ID]
				switch value.Name {
				case "read_file":
					path := sanitizePathForContext(stringArgument(args, "file_path", "path"))
					if path == "" {
						path = "unknown"
					}
					value.Content = []string{fmt.Sprintf("File content from %s. Re-read if needed.", path)}
					req.Events[eventIdx].Content[contentIdx] = value
				case "search_web", "tavily_search":
					query := sanitizeInlineForContext(stringArgument(args, "query"), 200)
					if query == "" {
						query = "unknown"
					}
					value.Content = []string{fmt.Sprintf("Search results processed. Query: %s. Re-search if needed.", query)}
					req.Events[eventIdx].Content[contentIdx] = value
				}
			}
		}
	}
	return nil
}

// ContextOptimizer applies a list of optimization strategies once a token
// threshold has been exceeded.
type ContextOptimizer struct {
	counter    TokenCounter
	threshold  int
	strategies []OptimizationStrategy
	logger     *slog.Logger
}

// NewContextOptimizer builds a request mutator that conditionally applies optimization strategies.
func NewContextOptimizer(counter TokenCounter, threshold int, strategies ...OptimizationStrategy) *ContextOptimizer {
	return &ContextOptimizer{counter: counter, threshold: threshold, strategies: slices.Clone(strategies)}
}

// WithLogger attaches structured optimization logs.
func (o *ContextOptimizer) WithLogger(logger *slog.Logger) *ContextOptimizer {
	o.logger = logger
	return o
}

// Mutate applies optimization strategies when the request exceeds the configured threshold.
func (o *ContextOptimizer) Mutate(ctx context.Context, req *model.Request) error {
	if o.counter == nil || o.threshold <= 0 {
		return nil
	}
	logDebug(o.logger, "context_optimize_start", "event_count", len(req.Events), "threshold", o.threshold)
	tokens, err := o.counter.Count(*req)
	if err != nil {
		logError(o.logger, "context_token_count_failed", "err", err)
		return fmt.Errorf("count tokens: %w", err)
	}
	logDebug(o.logger, "context_token_count", "tokens", tokens, "threshold", o.threshold)
	if tokens < o.threshold {
		logDebug(o.logger, "context_optimize_skipped", "tokens", tokens, "threshold", o.threshold)
		return nil
	}
	for _, strategy := range o.strategies {
		strategyName := typeName(strategy)
		startedAt := time.Now()
		logDebug(o.logger, "context_strategy_apply", "strategy", strategyName, "tokens_before", tokens)
		if err := strategy.Optimize(ctx, req); err != nil {
			logError(o.logger, "context_strategy_failed", "strategy", strategyName, "err", err)
			return err
		}
		afterTokens, err := o.counter.Count(*req)
		if err != nil {
			logError(o.logger, "context_token_recount_failed", "strategy", strategyName, "err", err)
			return fmt.Errorf("count tokens after optimization: %w", err)
		}
		logDebug(
			o.logger,
			"context_strategy_applied",
			"strategy", strategyName,
			"tokens_before", tokens,
			"tokens_after", afterTokens,
			"duration_ms", time.Since(startedAt).Milliseconds(),
		)
		tokens = afterTokens
		if tokens < o.threshold {
			return nil
		}
	}
	return nil
}

// SummarizationStrategy replaces older middle-history events with a generated summary.
type SummarizationStrategy struct {
	generator  SummaryGenerator
	keepRecent int
}

// NewSummarizationStrategy creates a summary-based optimization strategy.
func NewSummarizationStrategy(generator SummaryGenerator, keepRecent int) SummarizationStrategy {
	return SummarizationStrategy{generator: generator, keepRecent: keepRecent}
}

// Optimize inserts a summary into the instructions and removes summarized events.
func (s SummarizationStrategy) Optimize(ctx context.Context, req *model.Request) error {
	if s.generator == nil || s.keepRecent <= 0 || len(req.Events) == 0 {
		return nil
	}
	userIdx := -1
	for i, event := range req.Events {
		if event.Author != "user" {
			continue
		}
		for _, item := range event.Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" {
				userIdx = i
				break
			}
		}
		if userIdx >= 0 {
			break
		}
	}
	if userIdx < 0 {
		return nil
	}
	summaryStart := userIdx + 1
	summaryEnd := len(req.Events) - s.keepRecent
	if summaryEnd <= summaryStart {
		return nil
	}
	summary, err := s.generator.Summarize(ctx, append([]model.Event(nil), req.Events[summaryStart:summaryEnd]...))
	if err != nil {
		return fmt.Errorf("generate summary: %w", err)
	}
	summary = sanitizeBlockForContext(summary, 1200)
	if summary == "" {
		return nil
	}
	if !containsSection(req.Instructions, "[Previous work summary]") {
		if req.Instructions != "" && req.Instructions[len(req.Instructions)-1] != '\n' {
			req.Instructions += "\n\n"
		}
		req.Instructions += "[Previous work summary]\n" + summary
	}
	req.Events = append(append([]model.Event(nil), req.Events[:userIdx+1]...), req.Events[summaryEnd:]...)
	return nil
}

func cloneRequest(req model.Request) model.Request {
	cloned := req
	if len(req.Events) > 0 {
		cloned.Events = make([]model.Event, len(req.Events))
		for i, event := range req.Events {
			cloned.Events[i] = event
			if len(event.Content) > 0 {
				cloned.Events[i].Content = append([]model.ContentItem(nil), event.Content...)
			}
		}
	}
	if len(req.Tools) > 0 {
		cloned.Tools = slices.Clone(req.Tools)
	}
	return cloned
}

func decodeArguments(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var args map[string]any
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	return args, nil
}

func stringArgument(args map[string]any, keys ...string) string {
	for _, key := range keys {
		if text, ok := args[key].(string); ok {
			return text
		}
	}
	return ""
}

func lastUserMessage(events []model.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		for _, item := range events[i].Content {
			msg, ok := item.(model.Message)
			if ok && msg.Role == "user" {
				return msg.Content
			}
		}
	}
	return ""
}
