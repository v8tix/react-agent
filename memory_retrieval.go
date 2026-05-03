package agent

import "context"

// RetrievalCandidate is a normalized shortlist item that lets lexical,
// semantic, or hybrid retrieval stages speak the same shape before reranking.
type RetrievalCandidate struct {
	ID       string
	Content  string
	Metadata map[string]string
	Score    float64
}

// HybridRetriever returns a ranked candidate list for a query.
//
// Implementations commonly combine lexical search, semantic search, metadata
// filters, or any other retrieval signal behind one method.
type HybridRetriever interface {
	Retrieve(context.Context, string, int) ([]RetrievalCandidate, error)
}

// Reranker makes a second, usually more precise pass over an existing
// candidate set for a query.
//
// A common pattern is "retrieve 20 quickly, rerank to 5 carefully".
type Reranker interface {
	Rerank(context.Context, string, []RetrievalCandidate, int) ([]RetrievalCandidate, error)
}

// ChunkContextEnricher adds source-level context to a chunk before downstream
// indexing or retrieval so the chunk still makes sense on its own.
//
// Typical enrichments include document title, URL, section heading, or other
// metadata that would be lost if the raw chunk were indexed by itself.
type ChunkContextEnricher interface {
	EnrichChunk(context.Context, string, string, map[string]string) (string, error)
}
