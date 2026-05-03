package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/v8tix/react-agent/model"
)

const stateSuspendedRunKey = "memory.suspended_run"

// Session stores the persisted conversation history and runner state for one
// user-facing conversation.
type Session struct {
	SessionID string
	UserID    string
	Events    []model.Event
	State     map[string]any
	CreatedAt time.Time
	UpdatedAt time.Time
}

// SessionManager persists and reloads sessions for [SessionRunner].
type SessionManager interface {
	Create(sessionID, userID string) (*Session, error)
	Get(sessionID string) (*Session, error)
	Save(session *Session) error
	GetOrCreate(sessionID, userID string) (*Session, error)
}

// SessionPersister stores raw session snapshots for durable runners.
type SessionPersister interface {
	SaveSession(context.Context, Session) error
	LoadSession(context.Context, string) (Session, error)
}

// InMemorySessionManager is a thread-safe in-process [SessionManager].
type InMemorySessionManager struct {
	mu       sync.RWMutex
	sessions map[string]*Session
}

type persistedSessionManager struct {
	persister SessionPersister
}

type inMemorySessionPersister struct {
	mu       sync.RWMutex
	sessions map[string]Session
}

// NewInMemorySessionManager creates an empty in-memory session store.
func NewInMemorySessionManager() *InMemorySessionManager {
	return &InMemorySessionManager{sessions: map[string]*Session{}}
}

// NewPersistedSessionManager adapts a SessionPersister to the SessionManager interface.
func NewPersistedSessionManager(persister SessionPersister) SessionManager {
	return &persistedSessionManager{persister: persister}
}

func newInMemorySessionPersister() *inMemorySessionPersister {
	return &inMemorySessionPersister{sessions: map[string]Session{}}
}

// Create inserts a new session for the given session and user identifiers.
func (m *InMemorySessionManager) Create(sessionID, userID string) (*Session, error) {
	if userID == "" {
		return nil, fmt.Errorf("userID is required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.sessions[sessionID]; exists {
		return nil, fmt.Errorf("session %s already exists", sessionID)
	}
	now := time.Now()
	session := &Session{SessionID: sessionID, UserID: userID, Events: []model.Event{}, State: map[string]any{}, CreatedAt: now, UpdatedAt: now}
	m.sessions[sessionID] = cloneSession(session)
	return cloneSession(session), nil
}

// Get loads a previously saved session, or nil when it does not exist.
func (m *InMemorySessionManager) Get(sessionID string) (*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	session := m.sessions[sessionID]
	if session == nil {
		return nil, nil
	}
	return cloneSession(session), nil
}

// Save upserts the supplied session snapshot.
func (m *InMemorySessionManager) Save(session *Session) error {
	if session == nil {
		return fmt.Errorf("cannot save nil session")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	session.UpdatedAt = time.Now()
	m.sessions[session.SessionID] = cloneSession(session)
	return nil
}

// GetOrCreate returns an existing session for the same user or creates one.
func (m *InMemorySessionManager) GetOrCreate(sessionID, userID string) (*Session, error) {
	if userID == "" {
		return nil, fmt.Errorf("userID is required")
	}
	session, err := m.Get(sessionID)
	if err != nil {
		return nil, err
	}
	if session != nil {
		if session.UserID != userID {
			return nil, fmt.Errorf("session %s belongs to a different user", sessionID)
		}
		return session, nil
	}
	return m.Create(sessionID, userID)
}

func (m *persistedSessionManager) Create(sessionID, userID string) (*Session, error) {
	if userID == "" {
		return nil, fmt.Errorf("userID is required")
	}
	existing, err := m.Get(sessionID)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		return nil, fmt.Errorf("session %s already exists", sessionID)
	}
	now := time.Now()
	session := &Session{SessionID: sessionID, UserID: userID, Events: []model.Event{}, State: map[string]any{}, CreatedAt: now, UpdatedAt: now}
	if err := m.Save(session); err != nil {
		return nil, err
	}
	return cloneSession(session), nil
}

func (m *persistedSessionManager) Get(sessionID string) (*Session, error) {
	if m.persister == nil {
		return nil, nil
	}
	session, err := m.persister.LoadSession(context.Background(), sessionID)
	if err != nil {
		return nil, err
	}
	if session.SessionID == "" {
		return nil, nil
	}
	return cloneSession(&session), nil
}

func (m *persistedSessionManager) Save(session *Session) error {
	if session == nil {
		return fmt.Errorf("cannot save nil session")
	}
	if m.persister == nil {
		return fmt.Errorf("session persister is required")
	}
	session.UpdatedAt = time.Now()
	return m.persister.SaveSession(context.Background(), *cloneSession(session))
}

func (m *persistedSessionManager) GetOrCreate(sessionID, userID string) (*Session, error) {
	if userID == "" {
		return nil, fmt.Errorf("userID is required")
	}
	session, err := m.Get(sessionID)
	if err != nil {
		return nil, err
	}
	if session != nil {
		if session.UserID != userID {
			return nil, fmt.Errorf("session %s belongs to a different user", sessionID)
		}
		return session, nil
	}
	return m.Create(sessionID, userID)
}

func (p *inMemorySessionPersister) SaveSession(_ context.Context, session Session) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sessions[session.SessionID] = *cloneSession(&session)
	return nil
}

func (p *inMemorySessionPersister) LoadSession(_ context.Context, sessionID string) (Session, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	session, ok := p.sessions[sessionID]
	if !ok {
		return Session{}, nil
	}
	return *cloneSession(&session), nil
}

// RunStatus reports whether a session run finished or is waiting for input.
type RunStatus string

const (
	// StatusComplete indicates the session run finished with a final answer.
	StatusComplete RunStatus = "complete"
	// StatusPending indicates the run suspended awaiting external input.
	StatusPending RunStatus = "pending"
)

// RunResult summarizes one [SessionRunner] invocation.
type RunResult struct {
	Output             string
	ToolCalled         bool
	Status             RunStatus
	SessionID          string
	PendingInteraction *InteractionRequest
}

// SessionRunner replays prior events from a session, executes the agent loop,
// and persists the updated state after each run or resume so conversations can
// continue across separate calls.
type SessionRunner struct {
	agent    *Agent
	sessions SessionManager
	maxSteps int
	logger   *slog.Logger
}

// NewSessionRunner builds a session-aware wrapper around [Agent] for chat-style
// or workflow-style conversations that span multiple turns.
func NewSessionRunner(agent *Agent, sessions SessionManager, maxSteps int) *SessionRunner {
	if maxSteps <= 0 {
		maxSteps = 10
	}
	return &SessionRunner{agent: agent, sessions: sessions, maxSteps: maxSteps}
}

// WithLogger attaches structured lifecycle logging to the runner.
func (r *SessionRunner) WithLogger(logger *slog.Logger) *SessionRunner {
	r.logger = logger
	return r
}

// Run appends the new user input to the stored conversation, executes until the
// run completes or suspends, and then persists the updated session state.
func (r *SessionRunner) Run(ctx context.Context, sessionID, userID, userInput string) (*RunResult, error) {
	startedAt := time.Now()
	logInfo(r.logger, "session_run_start", "session_id", sessionID, "user_id", userID)
	session, err := r.getOrCreateSession(sessionID, userID)
	if err != nil {
		logError(r.logger, "session_run_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	execCtx := NewExecutionContextForTest()
	replayEvents(execCtx, session.Events)
	if userInput != "" {
		execCtx.AddEvent("user", model.Message{Role: "user", Content: userInput})
	}
	result, suspended, err := r.execute(ctx, execCtx)
	if err != nil {
		logError(r.logger, "session_run_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	persistExecutionState(session, execCtx, suspended)
	if err := r.sessions.Save(session); err != nil {
		logError(r.logger, "session_run_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	result.SessionID = sessionID
	logInfo(r.logger, "session_run_end", "session_id", sessionID, "user_id", userID, "status", result.Status, "duration_ms", time.Since(startedAt).Milliseconds())
	return result, nil
}

// Resume continues a previously suspended session using an external response.
func (r *SessionRunner) Resume(ctx context.Context, sessionID, userID string, response InteractionResponse) (*RunResult, error) {
	startedAt := time.Now()
	logInfo(r.logger, "session_resume_start", "session_id", sessionID, "user_id", userID, "request_id", response.RequestID)
	session, err := r.getOrCreateSession(sessionID, userID)
	if err != nil {
		logError(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	suspended, err := suspendedRunFromSession(session)
	if err != nil {
		logError(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	result, _, err := r.agent.Resume(ctx, suspended, response)
	if err != nil {
		var suspendedErr *InteractionRequestedError
		if errors.As(err, &suspendedErr) {
			persistExecutionState(session, suspendedErr.Suspended.Context, &suspendedErr.Suspended)
			if saveErr := r.sessions.Save(session); saveErr != nil {
				logError(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", saveErr)
				return nil, saveErr
			}
			logInfo(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "status", StatusPending, "duration_ms", time.Since(startedAt).Milliseconds())
			return &RunResult{Status: StatusPending, SessionID: sessionID, PendingInteraction: &suspendedErr.Suspended.Interaction}, nil
		}
		logError(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	clearSuspendedRun(session)
	session.Events = result.Context.Events()
	if err := r.sessions.Save(session); err != nil {
		logError(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "duration_ms", time.Since(startedAt).Milliseconds(), "err", err)
		return nil, err
	}
	logInfo(r.logger, "session_resume_end", "session_id", sessionID, "user_id", userID, "status", StatusComplete, "duration_ms", time.Since(startedAt).Milliseconds())
	return &RunResult{Output: result.Output, ToolCalled: result.ToolCalled, Status: StatusComplete, SessionID: sessionID}, nil
}

func (r *SessionRunner) execute(ctx context.Context, execCtx *ExecutionContext) (*RunResult, *SuspendedRun, error) {
	for execCtx.CurrentStep() < r.maxSteps {
		err := r.agent.Step(ctx, execCtx)
		if err != nil {
			var suspendedErr *InteractionRequestedError
			if errors.As(err, &suspendedErr) {
				return &RunResult{Status: StatusPending, PendingInteraction: &suspendedErr.Suspended.Interaction}, &suspendedErr.Suspended, nil
			}
			return nil, nil, err
		}
		if execCtx.Done() {
			output, _ := execCtx.FinalResult()
			return &RunResult{Output: output, ToolCalled: toolCalled(execCtx.Events()), Status: StatusComplete}, nil, nil
		}
		execCtx.IncrementStep()
	}
	return nil, nil, fmt.Errorf("session runner: max steps reached")
}

func replayEvents(execCtx *ExecutionContext, events []model.Event) {
	for _, event := range events {
		execCtx.AddEvent(event.Author, event.Content...)
	}
}

func persistExecutionState(session *Session, execCtx *ExecutionContext, suspended *SuspendedRun) {
	session.Events = execCtx.Events()
	if session.State == nil {
		session.State = map[string]any{}
	}
	if suspended == nil {
		delete(session.State, stateSuspendedRunKey)
		return
	}
	session.State[stateSuspendedRunKey] = *suspended
}

func clearSuspendedRun(session *Session) {
	if session.State != nil {
		delete(session.State, stateSuspendedRunKey)
	}
}

func suspendedRunFromSession(session *Session) (SuspendedRun, error) {
	if session.State == nil {
		return SuspendedRun{}, fmt.Errorf("session %s has no suspended run", session.SessionID)
	}
	raw, ok := session.State[stateSuspendedRunKey]
	if !ok {
		return SuspendedRun{}, fmt.Errorf("session %s has no suspended run", session.SessionID)
	}
	switch value := raw.(type) {
	case SuspendedRun:
		if value.Context == nil {
			break
		}
		return value, nil
	case *SuspendedRun:
		if value == nil || value.Context == nil {
			break
		}
		return *value, nil
	}
	return SuspendedRun{}, fmt.Errorf("session %s has invalid suspended run state", session.SessionID)
}

func (r *SessionRunner) getOrCreateSession(sessionID, userID string) (*Session, error) {
	if sessionID == "" {
		return nil, fmt.Errorf("sessionID is required")
	}
	if userID == "" {
		return nil, fmt.Errorf("userID is required")
	}
	return r.sessions.GetOrCreate(sessionID, userID)
}

func cloneSession(session *Session) *Session {
	if session == nil {
		return nil
	}
	cloned := *session
	if session.Events != nil {
		cloned.Events = make([]model.Event, len(session.Events))
		for i, event := range session.Events {
			cloned.Events[i] = event
			if len(event.Content) > 0 {
				cloned.Events[i].Content = append([]model.ContentItem(nil), event.Content...)
			}
		}
	}
	if session.State != nil {
		cloned.State = make(map[string]any, len(session.State))
		for key, value := range session.State {
			cloned.State[key] = value
		}
	}
	return &cloned
}

func toolCalled(events []model.Event) bool {
	for _, event := range events {
		if event.Author != "agent" {
			continue
		}
		for _, item := range event.Content {
			if _, ok := item.(model.ToolCall); ok {
				return true
			}
		}
	}
	return false
}
