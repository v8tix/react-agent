package templatecache

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"strings"
	"text/template"
)

const errWrapFormat = "%w: %v"

var (
	templateCache = make(map[string]*template.Template)

	ErrTemplateNotFound  = errors.New("error template not found")
	ErrWalkingDir        = errors.New("error walking directory")
	ErrReadingContent    = errors.New("error reading template content")
	ErrParsingTemplate   = errors.New("error parsing template")
	ErrGettingTemplate   = errors.New("error getting template")
	ErrExecutingTemplate = errors.New("error executing template")

	fsWalkDirFunc  = fs.WalkDir
	fsReadFileFunc = func(fsys fs.FS, name string) ([]byte, error) {
		return fs.ReadFile(fsys, name)
	}
	executeTemplateFunc = executeTemplateImpl
)

// PreloadTemplates loads all embedded templates into the cache.
// It parses all templates together to support template composition.
func PreloadTemplates(templates embed.FS, funcMap template.FuncMap) error {
	loadedTemplates, err := LoadTemplates(templates)
	if err != nil {
		return err
	}

	tmpl := template.New("").Funcs(funcMap)
	for path, content := range loadedTemplates {
		_, err := tmpl.New(path).Parse(string(content))
		if err != nil {
			return fmt.Errorf("%w: %v (template: %s)", ErrParsingTemplate, err, path)
		}
	}

	for path := range loadedTemplates {
		if namedTmpl := tmpl.Lookup(path); namedTmpl != nil {
			templateCache[path] = namedTmpl
		}
	}

	return nil
}

// LoadTemplates loads all embedded .gotmpl files into memory.
func LoadTemplates(templates embed.FS) (map[string][]byte, error) {
	templatesMap := make(map[string][]byte)

	err := fsWalkDirFunc(templates, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return fmt.Errorf(errWrapFormat, ErrWalkingDir, err)
		}

		if !d.IsDir() && strings.HasSuffix(path, ".gotmpl") {
			templateData, err := fsReadFileFunc(templates, path)
			if err != nil {
				return fmt.Errorf(errWrapFormat, ErrReadingContent, err)
			}
			templatesMap[path] = templateData
		}

		return nil
	})
	if err != nil {
		return templatesMap, fmt.Errorf(errWrapFormat, ErrWalkingDir, err)
	}

	return templatesMap, nil
}

// ExecuteTemplate executes a cached template with the provided data.
func ExecuteTemplate(templatePath string, data any) (string, error) {
	return executeTemplateFunc(templatePath, data)
}

func executeTemplateImpl(templatePath string, data any) (string, error) {
	tmpl, err := getTemplate(templatePath)
	if err != nil {
		return "", fmt.Errorf(errWrapFormat, ErrGettingTemplate, err)
	}

	var result strings.Builder
	if err := tmpl.Execute(&result, data); err != nil {
		return "", fmt.Errorf(errWrapFormat, ErrExecutingTemplate, err)
	}

	return result.String(), nil
}

func getTemplate(templatePath string) (*template.Template, error) {
	if cachedTemplate, ok := templateCache[templatePath]; ok {
		return cachedTemplate, nil
	}
	return nil, fmt.Errorf("%w: %s", ErrTemplateNotFound, templatePath)
}
