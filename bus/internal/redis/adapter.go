// SAHA – Redis Adapter for the Go Event Bus
// Implements bus.Adapter using Redis pub/sub.
// Replace this file with kafka_adapter.go when scaling to Phase 2.
package redis

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/redis/go-redis/v9"
)

// Adapter wraps go-redis in the bus.Adapter interface.
type Adapter struct {
	client *redis.Client
	log    *slog.Logger
}

// New creates a Redis Adapter connected to the given URL.
func New(redisURL string, log *slog.Logger) (*Adapter, error) {
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("redis: parse URL: %w", err)
	}
	client := redis.NewClient(opts)

	// Connectivity check
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis: ping failed: %w", err)
	}
	log.Info("redis adapter connected", "url", redisURL)
	return &Adapter{client: client, log: log}, nil
}

// Publish sends data bytes to a Redis channel (topic).
func (a *Adapter) Publish(ctx context.Context, topic string, data []byte) error {
	return a.client.Publish(ctx, topic, data).Err()
}

// Subscribe starts a goroutine that consumes messages from a Redis channel.
// handler is called synchronously within the goroutine; callers should
// make it non-blocking (e.g., dispatch to a worker pool).
func (a *Adapter) Subscribe(ctx context.Context, topic string, handler func([]byte)) error {
	ps := a.client.Subscribe(ctx, topic)
	ch := ps.Channel()

	go func() {
		defer ps.Close()
		for {
			select {
			case <-ctx.Done():
				a.log.Info("redis subscription closed", "topic", topic)
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				handler([]byte(msg.Payload))
			}
		}
	}()

	return nil
}

// Close shuts down the Redis client connection.
func (a *Adapter) Close() error {
	return a.client.Close()
}

// Ping checks Redis connectivity; used by the health endpoint.
func (a *Adapter) Ping(ctx context.Context) error {
	return a.client.Ping(ctx).Err()
}
