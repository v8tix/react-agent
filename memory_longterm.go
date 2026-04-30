package agent

import (
	"context"
	"crypto/rand"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/v8tix/react-agent/model"
)

// TaskMemory stores a reusable record of how a prior task was solved.
type TaskMemory struct {
	TaskSummary   string
	Approach      string
	FinalAnswer   string
	IsCorrect     bool
	ErrorAnalysis string
}

// EmbeddingText returns the text used to embed this memory for similarity search.
func (m TaskMemory) EmbeddingText() string { return "Task: " + m.TaskSummary }

// Embedder converts one or more texts into vectors for semantic search.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float64, error)
}

// VectorDocument binds an embedding to its original task memory payload.
type VectorDocument struct {
	ID     string
	Vector []float64
	Memory TaskMemory
}

// VectorStore persists vectorized memories and supports nearest-neighbor lookup.
type VectorStore interface {
	Add(ctx context.Context, docs []VectorDocument) error
	Search(ctx context.Context, query []float64, topK int) ([]VectorDocument, error)
}

// DuplicateChecker decides whether a memory is already represented in the store.
type DuplicateChecker interface {
	IsDuplicate(memory TaskMemory, existing []TaskMemory) bool
}

// SimpleDuplicateChecker treats an identical TaskMemory payload as a duplicate.
type SimpleDuplicateChecker struct{}

// IsDuplicate reports whether memory matches any candidate exactly.
func (SimpleDuplicateChecker) IsDuplicate(memory TaskMemory, existing []TaskMemory) bool {
	for _, candidate := range existing {
		if candidate == memory {
			return true
		}
	}
	return false
}

// InMemoryVectorStore is a thread-safe in-process vector store.
type InMemoryVectorStore struct {
	mu   sync.RWMutex
	docs []VectorDocument
}

// NewInMemoryVectorStore creates an empty in-memory vector store.
func NewInMemoryVectorStore() *InMemoryVectorStore { return &InMemoryVectorStore{} }

// Add appends new vector documents to the store.
func (s *InMemoryVectorStore) Add(_ context.Context, docs []VectorDocument) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs = append(s.docs, docs...)
	return nil
}

// Search returns the top-K documents ranked by cosine similarity.
func (s *InMemoryVectorStore) Search(_ context.Context, query []float64, topK int) ([]VectorDocument, error) {
	if topK <= 0 {
		return nil, fmt.Errorf("topK must be greater than zero")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	type scored struct {
		doc   VectorDocument
		score float64
	}
	scoredDocs := make([]scored, 0, len(s.docs))
	for _, doc := range s.docs {
		scoredDocs = append(scoredDocs, scored{doc: doc, score: cosineSimilarity(query, doc.Vector)})
	}
	slices.SortFunc(scoredDocs, func(a, b scored) int {
		switch {
		case a.score > b.score:
			return -1
		case a.score < b.score:
			return 1
		default:
			return 0
		}
	})
	if topK > len(scoredDocs) {
		topK = len(scoredDocs)
	}
	out := make([]VectorDocument, 0, topK)
	for _, item := range scoredDocs[:topK] {
		out = append(out, item.doc)
	}
	return out, nil
}

// TaskMemoryManager saves and retrieves semantically indexed task memories.
type TaskMemoryManager struct {
	embedder         Embedder
	store            VectorStore
	duplicateChecker DuplicateChecker
	logger           *slog.Logger
}

// NewTaskMemoryManager creates a semantic memory manager from its pluggable components.
func NewTaskMemoryManager(embedder Embedder, store VectorStore, duplicateChecker DuplicateChecker) *TaskMemoryManager {
	return &TaskMemoryManager{embedder: embedder, store: store, duplicateChecker: duplicateChecker}
}

// WithLogger attaches structured save and search logs.
func (m *TaskMemoryManager) WithLogger(logger *slog.Logger) *TaskMemoryManager {
	m.logger = logger
	return m
}

// Save embeds, de-duplicates, and persists a task memory.
func (m *TaskMemoryManager) Save(ctx context.Context, memory TaskMemory) (string, bool, error) {
	logInfo(m.logger, "memory_save_start", "task_summary", sanitizeInlineForContext(memory.TaskSummary, 120))
	memory = sanitizeTaskMemory(memory)
	vector, err := m.embedOne(ctx, memory.EmbeddingText())
	if err != nil {
		logError(m.logger, "memory_save_end", "saved", false, "err", err)
		return "", false, err
	}
	existingDocs, err := m.store.Search(ctx, vector, 3)
	if err != nil {
		logError(m.logger, "memory_save_end", "saved", false, "err", err)
		return "", false, err
	}
	existing := make([]TaskMemory, 0, len(existingDocs))
	for _, doc := range existingDocs {
		existing = append(existing, doc.Memory)
	}
	if m.duplicateChecker != nil && m.duplicateChecker.IsDuplicate(memory, existing) {
		logInfo(m.logger, "memory_save_end", "saved", false, "duplicate", true)
		return "", false, nil
	}
	id, err := newMemoryID()
	if err != nil {
		logError(m.logger, "memory_save_end", "saved", false, "err", err)
		return "", false, err
	}
	if err := m.store.Add(ctx, []VectorDocument{{ID: id, Vector: vector, Memory: memory}}); err != nil {
		logError(m.logger, "memory_save_end", "saved", false, "err", err)
		return "", false, err
	}
	logInfo(m.logger, "memory_save_end", "saved", true, "memory_id", id)
	return id, true, nil
}

// Search retrieves similar task memories for a natural-language query.
func (m *TaskMemoryManager) Search(ctx context.Context, query string, topK int) ([]TaskMemory, error) {
	query = sanitizeInlineForContext(query, 240)
	logDebug(m.logger, "memory_search_start", "query", query, "top_k", topK)
	vector, err := m.embedOne(ctx, query)
	if err != nil {
		logError(m.logger, "memory_search_end", "top_k", topK, "err", err)
		return nil, err
	}
	docs, err := m.store.Search(ctx, vector, topK)
	if err != nil {
		logError(m.logger, "memory_search_end", "top_k", topK, "err", err)
		return nil, err
	}
	memories := make([]TaskMemory, 0, len(docs))
	for _, doc := range docs {
		memories = append(memories, sanitizeTaskMemory(doc.Memory))
	}
	logDebug(m.logger, "memory_search_end", "top_k", topK, "results", len(memories))
	return memories, nil
}

func (m *TaskMemoryManager) embedOne(ctx context.Context, text string) ([]float64, error) {
	vectors, err := m.embedder.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("expected 1 embedding, got %d", len(vectors))
	}
	return vectors[0], nil
}

// MemorySearcher retrieves similar task memories for prompt injection.
type MemorySearcher interface {
	Search(ctx context.Context, query string, topK int) ([]TaskMemory, error)
}

// MemoryInjector adds retrieved long-term memories to the prompt instructions.
type MemoryInjector struct {
	searcher MemorySearcher
	topK     int
}

// NewMemoryInjector creates a request mutator backed by semantic memory search.
func NewMemoryInjector(searcher MemorySearcher, topK int) MemoryInjector {
	if topK <= 0 {
		topK = 3
	}
	return MemoryInjector{searcher: searcher, topK: topK}
}

// Mutate looks up similar past tasks and injects them into the instructions.
func (i MemoryInjector) Mutate(ctx context.Context, req *model.Request) error {
	if i.searcher == nil {
		return nil
	}
	query := sanitizeInlineForContext(lastUserMessage(req.Events), 240)
	if query == "" {
		return nil
	}
	memories, err := i.searcher.Search(ctx, query, i.topK)
	if err != nil {
		return err
	}
	if len(memories) == 0 || containsSection(req.Instructions, "<PAST_EXPERIENCES>") {
		return nil
	}
	if req.Instructions != "" {
		req.Instructions += "\n\n"
	}
	req.Instructions += "The following are records from similar problems solved in the past:\n<PAST_EXPERIENCES>\n" + formatMemories(memories) + "\n</PAST_EXPERIENCES>\nReference successful approaches and avoid approaches that led to failures."
	return nil
}

func formatMemories(memories []TaskMemory) string {
	lines := make([]string, 0, len(memories))
	for idx, memory := range memories {
		memory = sanitizeTaskMemory(memory)
		status := "Incorrect"
		if memory.IsCorrect {
			status = "Correct"
		}
		line := fmt.Sprintf("[Record %d]\n- Problem: %s\n- Approach: %s\n- Answer: %s\n- Result: %s", idx+1, memory.TaskSummary, memory.Approach, memory.FinalAnswer, status)
		if memory.ErrorAnalysis != "" {
			line += "\n- Error analysis: " + memory.ErrorAnalysis
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n\n")
}

func sanitizeTaskMemory(memory TaskMemory) TaskMemory {
	memory.TaskSummary = sanitizeInlineForContext(memory.TaskSummary, 240)
	memory.Approach = sanitizeBlockForContext(memory.Approach, 600)
	memory.FinalAnswer = sanitizeInlineForContext(memory.FinalAnswer, 240)
	memory.ErrorAnalysis = sanitizeBlockForContext(memory.ErrorAnalysis, 400)
	return memory
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func newMemoryID() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("generate memory id: %w", err)
	}
	return fmt.Sprintf("mem-%x", buf), nil
}
