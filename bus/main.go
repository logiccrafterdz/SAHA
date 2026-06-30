// SAHA – Event Bus Service Entry Point
// Starts the Go bus daemon: connects to Redis, subscribes to all SAHA topics,
// and exposes an HTTP health endpoint on :8090.
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"saha-bus/internal/bus"
	redisadapter "saha-bus/internal/redis"
)

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	redisURL := getEnv("REDIS_URL", "redis://localhost:6379/0")
	httpPort := getEnv("BUS_PORT", "8090")

	// ── Connect Redis adapter ────────────────────────────────────────────────
	adapter, err := redisadapter.New(redisURL, log)
	if err != nil {
		log.Error("failed to connect redis", "err", err)
		os.Exit(1)
	}

	b := bus.New(adapter, log)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ── Subscribe to all SAHA topics (log-only handler for the Go bus) ───────
	// In Phase 1 the Python services communicate directly via Redis pub/sub.
	// The Go bus acts as the canonical broker: it can add routing, filtering,
	// replay, and dead-letter queues in Phase 2 without touching Python code.
	for _, topic := range bus.AllTopics {
		topicCopy := topic
		if err := b.Subscribe(ctx, topicCopy, func(_ context.Context, t string, payload map[string]any) {
			log.Info("bus event", "topic", t, "keys", mapKeys(payload))
		}); err != nil {
			log.Error("subscribe failed", "topic", topicCopy, "err", err)
			os.Exit(1)
		}
		log.Info("subscribed", "topic", topicCopy)
	}

	// ── HTTP health + metrics ────────────────────────────────────────────────
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	r.Get("/health", func(w http.ResponseWriter, req *http.Request) {
		if err := adapter.Ping(req.Context()); err != nil {
			http.Error(w, `{"status":"degraded","redis":false}`, http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok","service":"saha-bus","topics":` + topicsJSON() + `}`))
	})

	r.Get("/topics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(topicsJSON()))
	})

	srv := &http.Server{
		Addr:         ":" + httpPort,
		Handler:      r,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	go func() {
		log.Info("saha-bus HTTP listening", "port", httpPort)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("http server error", "err", err)
		}
	}()

	// ── Graceful shutdown ────────────────────────────────────────────────────
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("shutting down saha-bus...")
	cancel()
	shutCtx, shutCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutCancel()
	_ = srv.Shutdown(shutCtx)
	_ = b.Close()
	log.Info("saha-bus stopped")
}

func mapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func topicsJSON() string {
	out := `[`
	for i, t := range bus.AllTopics {
		if i > 0 {
			out += ","
		}
		out += `"` + t + `"`
	}
	return out + `]`
}
