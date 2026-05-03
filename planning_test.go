package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/v8tix/react-agent/model"
)

func TestPlanTask_StringFormatsByStatus(t *testing.T) {
	tests := []struct {
		name string
		task PlanTask
		want string
	}{
		{
			name: "pending",
			task: PlanTask{Content: "Find source", Status: PlanTaskPending},
			want: "[ ] Find source",
		},
		{
			name: "in progress",
			task: PlanTask{Content: "Calculate result", Status: PlanTaskInProgress},
			want: "[>] **Calculate result**",
		},
		{
			name: "completed",
			task: PlanTask{Content: "Search complete", Status: PlanTaskCompleted},
			want: "[x] ~~Search complete~~",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.task.String(); got != tt.want {
				t.Fatalf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFormatPlanTasks_JoinsFormattedTasks(t *testing.T) {
	got := FormatPlanTasks([]PlanTask{
		{Content: "Find Kipchoge marathon world record time", Status: PlanTaskCompleted},
		{Content: "Calculate pace in km/h", Status: PlanTaskInProgress},
		{Content: "Find Earth-Moon distance at perigee", Status: PlanTaskPending},
	})

	want := strings.Join([]string{
		"[x] ~~Find Kipchoge marathon world record time~~",
		"[>] **Calculate pace in km/h**",
		"[ ] Find Earth-Moon distance at perigee",
	}, "\n")
	if got != want {
		t.Fatalf("FormatPlanTasks() = %q, want %q", got, want)
	}
}

func TestPlanningToolDefinition_DescribesPlanningUsage(t *testing.T) {
	def := PlanningToolDefinition()

	if def.Name != "create_tasks" {
		t.Fatalf("Name = %q, want create_tasks", def.Name)
	}
	for _, want := range []string{
		"Complex queries requiring multiple steps of research",
		"Simple questions answerable with a single search",
		"Mark completed tasks as 'completed'",
	} {
		if !strings.Contains(def.Description, want) {
			t.Fatalf("Description missing %q: %s", want, def.Description)
		}
	}

	properties, ok := def.Parameters["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties missing or wrong type: %#v", def.Parameters["properties"])
	}
	tasks, ok := properties["tasks"].(map[string]any)
	if !ok {
		t.Fatalf("tasks schema missing: %#v", properties["tasks"])
	}
	if tasks["type"] != "array" {
		t.Fatalf("tasks type = %#v, want array", tasks["type"])
	}
}

func TestMarshalAndParsePlanTasks_RoundTrip(t *testing.T) {
	want := []PlanTask{
		{Content: "Find source", Status: PlanTaskPending},
		{Content: "Calculate result", Status: PlanTaskInProgress},
		{Content: "Verify answer", Status: PlanTaskCompleted},
	}

	raw, err := MarshalPlanTasks(want)
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	got, err := ParsePlanTasks(raw)
	if err != nil {
		t.Fatalf("ParsePlanTasks() error = %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("len(ParsePlanTasks()) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ParsePlanTasks()[%d] = %#v, want %#v", i, got[i], want[i])
		}
	}
}

func TestPlanningExecutor_CapturesPlanSnapshots(t *testing.T) {
	executor := NewPlanningExecutor(nil)
	raw, err := MarshalPlanTasks([]PlanTask{
		{Content: "Find source", Status: PlanTaskPending},
		{Content: "Calculate result", Status: PlanTaskInProgress},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	results, err := executor.Execute(context.Background(), []model.ToolCall{{
		ID:        "tc-1",
		Name:      "create_tasks",
		Arguments: raw,
	}})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].Status != "success" {
		t.Fatalf("results[0].Status = %q, want success", results[0].Status)
	}
	if len(executor.Plans()) != 1 {
		t.Fatalf("len(Plans()) = %d, want 1", len(executor.Plans()))
	}
	if got, ok := executor.LatestPlan(); !ok || !strings.Contains(got, "[>] **Calculate result**") {
		t.Fatalf("LatestPlan() = (%q, %v), want formatted plan with in-progress task", got, ok)
	}
	if counts := executor.TaskCounts(); len(counts) != 1 || counts[0] != 2 {
		t.Fatalf("TaskCounts() = %#v, want [2]", counts)
	}
}

func TestPlanningExecutor_DelegatesUnknownTools(t *testing.T) {
	delegate := &fakeToolExecutor{
		results: []model.ToolResult{{ID: "tc-2", Name: "search_web", Status: "success", Content: []string{"ok"}}},
	}
	executor := NewPlanningExecutor(delegate)

	results, err := executor.Execute(context.Background(), []model.ToolCall{{
		ID:        "tc-2",
		Name:      "search_web",
		Arguments: []byte(`{"query":"moon"}`),
	}})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if len(results) != 1 || results[0].Name != "search_web" {
		t.Fatalf("results = %#v, want delegated search_web result", results)
	}
	if delegate.calls != 1 {
		t.Fatalf("delegate calls = %d, want 1", delegate.calls)
	}
}

func TestPlanningExecutor_ObserversSeeRevisions(t *testing.T) {
	observer := &recordingPlanningObserver{}
	executor := NewPlanningExecutor(nil).WithObservers(observer)

	raw, err := MarshalPlanTasks([]PlanTask{
		{Content: "Find source", Status: PlanTaskCompleted},
		{Content: "Draft answer", Status: PlanTaskInProgress},
	})
	if err != nil {
		t.Fatalf("MarshalPlanTasks() error = %v", err)
	}

	_, err = executor.Execute(context.Background(), []model.ToolCall{{
		ID:        "tc-1",
		Name:      PlanningToolDefinition().Name,
		Arguments: raw,
	}})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if len(observer.revisions) != 1 {
		t.Fatalf("observer revisions = %d, want 1", len(observer.revisions))
	}
	if observer.revisions[0].TaskCount != 2 {
		t.Fatalf("observer revision = %#v, want task count 2", observer.revisions[0])
	}
}

func TestPlanningExecutor_RevisionsRemainReadableDuringRepeatedUpdates(t *testing.T) {
	executor := NewPlanningExecutor(nil)

	const updates = 25
	var wg sync.WaitGroup
	errCh := make(chan error, updates+1)

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < updates; i++ {
			revisions := executor.Revisions()
			if len(revisions) > updates {
				errCh <- context.Canceled
				return
			}
		}
	}()

	for i := 0; i < updates; i++ {
		raw, err := MarshalPlanTasks([]PlanTask{
			{Content: "Find source", Status: PlanTaskCompleted},
			{Content: "Draft answer", Status: PlanTaskInProgress},
		})
		if err != nil {
			t.Fatalf("MarshalPlanTasks() error = %v", err)
		}
		if _, err := executor.Execute(context.Background(), []model.ToolCall{{
			ID:        "tc-repeat",
			Name:      PlanningToolDefinition().Name,
			Arguments: raw,
		}}); err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent revision read failed: %v", err)
		}
	}
	if got := len(executor.Revisions()); got != updates {
		t.Fatalf("len(Revisions()) = %d, want %d", got, updates)
	}
}

type fakeToolExecutor struct {
	results []model.ToolResult
	calls   int
}

func (f *fakeToolExecutor) Execute(_ context.Context, _ []model.ToolCall) ([]model.ToolResult, error) {
	f.calls++
	return f.results, nil
}

type recordingPlanningObserver struct {
	revisions []PlanRevision
}

func (r *recordingPlanningObserver) OnPlanRevision(revision PlanRevision) {
	r.revisions = append(r.revisions, revision)
}
