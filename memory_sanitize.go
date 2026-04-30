package agent

import (
	"encoding/json"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

var unsafePromptPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)ignore\s+previous\s+instructions?`),
	regexp.MustCompile(`(?i)\[/?inst\]`),
	regexp.MustCompile(`(?i)<\|/?im_(start|end)\|>`),
}

var sensitiveArgumentKeys = []string{
	"password", "secret", "token", "api_key", "apikey", "authorization", "cookie", "credential",
}

func sanitizeBlockForContext(text string, maxLen int) string {
	sanitized := sanitizeCommon(text)
	if maxLen > 0 && len(sanitized) > maxLen {
		sanitized = sanitized[:maxLen] + "..."
	}
	return strings.TrimSpace(sanitized)
}

func sanitizeInlineForContext(text string, maxLen int) string {
	sanitized := sanitizeBlockForContext(text, maxLen)
	sanitized = strings.ReplaceAll(sanitized, "\n", " ")
	sanitized = strings.Join(strings.Fields(sanitized), " ")
	return sanitized
}

func sanitizePathForContext(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	clean := filepath.Clean(path)
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") || strings.Contains(clean, "/../") || strings.Contains(clean, `\..\`) {
		return "[redacted-path]"
	}
	return sanitizeInlineForContext(clean, 120)
}

func sanitizeCommon(text string) string {
	text = strings.Map(func(r rune) rune {
		if r == '\n' || r == '\t' {
			return r
		}
		if unicode.IsControl(r) {
			return ' '
		}
		return r
	}, text)
	for _, pattern := range unsafePromptPatterns {
		text = pattern.ReplaceAllString(text, "[filtered]")
	}
	return text
}

func containsSection(text, marker string) bool {
	return strings.Contains(text, marker)
}

func redactToolArguments(raw json.RawMessage) any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "[redacted-invalid-arguments]"
	}
	return redactValue("", decoded)
}

func redactValue(key string, value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for childKey, childValue := range typed {
			out[childKey] = redactValue(childKey, childValue)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = redactValue(key, item)
		}
		return out
	case string:
		lowerKey := strings.ToLower(key)
		for _, sensitive := range sensitiveArgumentKeys {
			if strings.Contains(lowerKey, sensitive) {
				return "[redacted]"
			}
		}
		if strings.Contains(lowerKey, "path") || strings.Contains(lowerKey, "file") || strings.Contains(lowerKey, "dir") || strings.Contains(lowerKey, "name") {
			return sanitizePathForContext(typed)
		}
		return sanitizeInlineForContext(typed, 160)
	default:
		return value
	}
}
