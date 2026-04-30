package agent

import (
	"context"
	"log/slog"
	"reflect"
	"time"

	"github.com/v8tix/react-agent/model"
)

func logDebug(logger *slog.Logger, msg string, args ...any) {
	if logger != nil {
		logger.Debug(msg, args...)
	}
}

func logInfo(logger *slog.Logger, msg string, args ...any) {
	if logger != nil {
		logger.Info(msg, args...)
	}
}

func logError(logger *slog.Logger, msg string, args ...any) {
	if logger != nil {
		logger.Error(msg, args...)
	}
}

func typeName(value any) string {
	if value == nil {
		return "<nil>"
	}
	typ := reflect.TypeOf(value)
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if typ.Name() != "" {
		return typ.Name()
	}
	return reflect.TypeOf(value).String()
}

type loggingRequestMutator struct {
	delegate RequestMutator
	logger   *slog.Logger
}

// WithMutatorLogger wraps a request mutator with structured start/finish logging.
func WithMutatorLogger(mutator RequestMutator, logger *slog.Logger) RequestMutator {
	if mutator == nil || logger == nil {
		return mutator
	}
	return loggingRequestMutator{delegate: mutator, logger: logger}
}

func (m loggingRequestMutator) Mutate(ctx context.Context, req *model.Request) error {
	startedAt := time.Now()
	name := typeName(m.delegate)
	logDebug(m.logger, "mutator_start", "mutator", name, "event_count", len(req.Events))
	err := m.delegate.Mutate(ctx, req)
	args := []any{"mutator", name, "duration_ms", time.Since(startedAt).Milliseconds()}
	if err != nil {
		args = append(args, "err", err)
		logError(m.logger, "mutator_finish", args...)
		return err
	}
	logDebug(m.logger, "mutator_finish", args...)
	return nil
}
