package templatecache

import (
	"embed"
	"strings"
	"testing"
)

//go:embed testdata/*.gotmpl
var testTemplates embed.FS

func TestPreloadTemplatesAndExecuteTemplate(t *testing.T) {
	if err := PreloadTemplates(testTemplates, nil); err != nil {
		t.Fatalf("PreloadTemplates() error = %v", err)
	}

	got, err := ExecuteTemplate("testdata/simple.gotmpl", map[string]any{
		"Name": "planning",
	})
	if err != nil {
		t.Fatalf("ExecuteTemplate() error = %v", err)
	}
	if strings.TrimSpace(got) != "hello planning" {
		t.Fatalf("ExecuteTemplate() = %q, want %q", got, "hello planning")
	}
}

func TestExecuteTemplate_MissingTemplate(t *testing.T) {
	_, err := ExecuteTemplate("missing.gotmpl", nil)
	if err == nil {
		t.Fatal("ExecuteTemplate() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), ErrTemplateNotFound.Error()) {
		t.Fatalf("ExecuteTemplate() error = %v, want contains %q", err, ErrTemplateNotFound)
	}
}
