package agent

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/v8tix/react-agent/internal/templatecache"
	"github.com/v8tix/react-agent/model"
)

type planningExecContextKey struct{}
type planningEventChannelKey struct{}

// PlanTaskStatus describes the execution state of a task in a generated plan.
type PlanTaskStatus string

const (
	// PlanTaskPending is used for tasks that have not started yet.
	PlanTaskPending PlanTaskStatus = "pending"
	// PlanTaskInProgress is used for the next task the agent should work on.
	PlanTaskInProgress PlanTaskStatus = "in_progress"
	// PlanTaskCompleted is used for tasks that are already finished.
	PlanTaskCompleted PlanTaskStatus = "completed"
)

// PlanTask represents a single item in a planning tool result.
type PlanTask struct {
	Content string
	Status  PlanTaskStatus
}

type planningToolArgs struct {
	Tasks []PlanTask `json:"tasks"`
}

// PlanRevision represents a captured planning snapshot.
type PlanRevision struct {
	Index     int
	Plan      string
	TaskCount int
}

// PlanningObserver receives each captured planning revision as it is recorded.
type PlanningObserver interface {
	OnPlanRevision(revision PlanRevision)
}

const planningDescriptionTemplatePath = "templates/planning_tool_description.gotmpl"

var (
	planningTemplatesOnce sync.Once
	planningTemplatesErr  error
)

//go:embed templates/*.gotmpl
var planningTemplates embed.FS

// String formats the task using the chapter's checklist-style representation.
func (t PlanTask) String() string {
	switch t.Status {
	case PlanTaskPending:
		return "[ ] " + t.Content
	case PlanTaskInProgress:
		return "[>] **" + t.Content + "**"
	case PlanTaskCompleted:
		return "[x] ~~" + t.Content + "~~"
	default:
		return t.Content
	}
}

// FormatPlanTasks renders a full plan in the format expected by the planning flow.
func FormatPlanTasks(tasks []PlanTask) string {
	lines := make([]string, len(tasks))
	for i, task := range tasks {
		lines[i] = task.String()
	}
	return strings.Join(lines, "\n")
}

// ParsePlanTasks unmarshals the JSON payload used by the create_tasks tool into
// the typed plan representation.
func ParsePlanTasks(data []byte) ([]PlanTask, error) {
	var payload planningToolArgs
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	return payload.Tasks, nil
}

// MarshalPlanTasks marshals plan tasks into the JSON payload expected by the
// create_tasks tool.
func MarshalPlanTasks(tasks []PlanTask) (json.RawMessage, error) {
	data, err := json.Marshal(planningToolArgs{Tasks: tasks})
	if err != nil {
		return nil, err
	}
	return json.RawMessage(data), nil
}

// PlanningExecutor executes create_tasks calls, returning the formatted plan
// and capturing each plan snapshot for later inspection. Unknown tools can be
// delegated to another executor when composition is needed.
type PlanningExecutor struct {
	delegate   model.ToolExecutor
	observers  []PlanningObserver
	mu         sync.Mutex
	plans      []string
	taskCounts []int
}

// NewPlanningExecutor creates a PlanningExecutor that optionally delegates
// non-planning tools to another executor.
func NewPlanningExecutor(delegate model.ToolExecutor) *PlanningExecutor {
	return &PlanningExecutor{delegate: delegate}
}

// WithObservers registers planning observers that are notified for each new
// captured revision.
func (e *PlanningExecutor) WithObservers(observers ...PlanningObserver) *PlanningExecutor {
	e.observers = append(e.observers, observers...)
	return e
}

// Execute handles planning tool calls directly and delegates all other calls to
// the underlying executor when present.
func (e *PlanningExecutor) Execute(ctx context.Context, calls []model.ToolCall) ([]model.ToolResult, error) {
	out := make([]model.ToolResult, len(calls))
	execCtx, _ := ctx.Value(planningExecContextKey{}).(*ExecutionContext)
	eventCh, _ := ctx.Value(planningEventChannelKey{}).(chan<- AgentEvent)
	for i, call := range calls {
		switch call.Name {
		case PlanningToolDefinition().Name:
			tasks, err := ParsePlanTasks(call.Arguments)
			if err != nil {
				return nil, err
			}
			formatted := FormatPlanTasks(tasks)
			revision := e.record(formatted, len(tasks))
			if execCtx != nil && eventCh != nil {
				emit(eventCh, PlanRevisionEvent{
					RunID:    execCtx.id,
					Step:     execCtx.currentStep,
					Revision: revision,
				})
			}
			out[i] = model.ToolResult{
				ID:      call.ID,
				Name:    call.Name,
				Status:  "success",
				Content: []string{formatted},
			}
		default:
			if e.delegate == nil {
				return nil, fmt.Errorf("planning executor: no delegate for tool %q", call.Name)
			}
			results, err := e.delegate.Execute(ctx, []model.ToolCall{call})
			if err != nil {
				return nil, err
			}
			if len(results) != 1 {
				return nil, fmt.Errorf("planning executor: delegate returned %d results for 1 call", len(results))
			}
			out[i] = results[0]
		}
	}
	return out, nil
}

// Plans returns the recorded formatted plan snapshots.
func (e *PlanningExecutor) Plans() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.plans...)
}

// TaskCounts returns the recorded task count for each captured plan snapshot.
func (e *PlanningExecutor) TaskCounts() []int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]int(nil), e.taskCounts...)
}

// LatestPlan returns the most recently captured plan snapshot.
func (e *PlanningExecutor) LatestPlan() (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.plans) == 0 {
		return "", false
	}
	return e.plans[len(e.plans)-1], true
}

// Revisions returns the recorded planning snapshots in a typed form that is
// easier to reuse in policies, tests, and observers.
func (e *PlanningExecutor) Revisions() []PlanRevision {
	e.mu.Lock()
	defer e.mu.Unlock()

	revisions := make([]PlanRevision, len(e.plans))
	for i := range e.plans {
		revisions[i] = PlanRevision{
			Index:     i,
			Plan:      e.plans[i],
			TaskCount: e.taskCounts[i],
		}
	}
	return revisions
}

func (e *PlanningExecutor) record(plan string, taskCount int) PlanRevision {
	e.mu.Lock()
	revision := PlanRevision{
		Index:     len(e.plans),
		Plan:      plan,
		TaskCount: taskCount,
	}
	e.plans = append(e.plans, plan)
	e.taskCounts = append(e.taskCounts, taskCount)
	observers := append([]PlanningObserver(nil), e.observers...)
	e.mu.Unlock()

	for _, observer := range observers {
		observer.OnPlanRevision(revision)
	}
	return revision
}

// PlanningToolDefinition returns a reusable ToolDefinition for Chapter 7 style planning.
func PlanningToolDefinition() model.ToolDefinition {
	return model.ToolDefinition{
		Name:        "create_tasks",
		Description: planningToolDescription(),
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"tasks": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"content": map[string]any{"type": "string"},
							"status": map[string]any{
								"type": "string",
								"enum": []string{
									string(PlanTaskPending),
									string(PlanTaskInProgress),
									string(PlanTaskCompleted),
								},
							},
						},
						"required": []string{"content", "status"},
					},
				},
			},
			"required": []string{"tasks"},
		},
	}
}

func planningToolDescription() string {
	planningTemplatesOnce.Do(func() {
		planningTemplatesErr = templatecache.PreloadTemplates(planningTemplates, nil)
	})
	if planningTemplatesErr != nil {
		panic(planningTemplatesErr)
	}

	description, err := templatecache.ExecuteTemplate(planningDescriptionTemplatePath, map[string]any{
		"Pending":    PlanTaskPending,
		"InProgress": PlanTaskInProgress,
		"Completed":  PlanTaskCompleted,
	})
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(description)
}
