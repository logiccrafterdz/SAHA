// SAHA – Core Bus logic (Go)
// Thin wrapper around the Redis adapter providing:
//   - Publish(topic, payload)
//   - Subscribe(topic, handler)
//   - Health check
// Designed to be adapter-agnostic: swap redis.Adapter for kafka.Adapter
// in main.go without touching this file.
package bus

import (
	"context"
	"encoding/json"
	"log/slog"
)

// MessageHandler is an async callback invoked for each received message.
type MessageHandler func(ctx context.Context, topic string, payload map[string]any)

// Adapter defines the pluggable backend interface.
type Adapter interface {
	Publish(ctx context.Context, topic string, data []byte) error
	Subscribe(ctx context.Context, topic string, handler func([]byte)) error
	Close() error
}

// Bus is the central SAHA event bus.
type Bus struct {
	adapter  Adapter
	handlers map[string][]MessageHandler
	log      *slog.Logger
}

// New creates a Bus with the provided adapter.
func New(adapter Adapter, log *slog.Logger) *Bus {
	return &Bus{
		adapter:  adapter,
		handlers: make(map[string][]MessageHandler),
		log:      log,
	}
}

// Publish serialises payload to JSON and sends it on topic.
func (b *Bus) Publish(ctx context.Context, topic string, payload map[string]any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	b.log.Debug("publishing", "topic", topic, "bytes", len(data))
	return b.adapter.Publish(ctx, topic, data)
}

// Subscribe registers a handler on a topic and starts consuming.
// Multiple handlers per topic are supported.
func (b *Bus) Subscribe(ctx context.Context, topic string, handler MessageHandler) error {
	b.handlers[topic] = append(b.handlers[topic], handler)

	return b.adapter.Subscribe(ctx, topic, func(data []byte) {
		var payload map[string]any
		if err := json.Unmarshal(data, &payload); err != nil {
			b.log.Warn("cannot unmarshal message", "topic", topic, "err", err)
			return
		}
		for _, h := range b.handlers[topic] {
			go h(ctx, topic, payload) // fire-and-forget per handler
		}
	})
}

// Close tears down the underlying adapter.
func (b *Bus) Close() error {
	return b.adapter.Close()
}
