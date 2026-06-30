-- SAHA Phase 1 – Initial PostgreSQL Schema
-- Uses JSONB columns for flexible contract storage (spec §3.2, §1.5, §2.4)

BEGIN;

-- ─── Extensions ────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- GIN indexes on JSONB

-- ─── provider_profiles ─────────────────────────────────────────────────────
-- Stores ProviderProfile (§2.4) for each known provider.
CREATE TABLE IF NOT EXISTS provider_profiles (
    provider_id   TEXT        PRIMARY KEY,
    profile       JSONB       NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_provider_profiles_profile
    ON provider_profiles USING GIN (profile);

-- ─── agent_states ───────────────────────────────────────────────────────────
-- Persists AgentState (§3.2) for every running/completed task.
CREATE TABLE IF NOT EXISTS agent_states (
    agent_state_id  TEXT        PRIMARY KEY,
    task_id         TEXT        NOT NULL,
    provider_id     TEXT        NOT NULL REFERENCES provider_profiles(provider_id),
    status          TEXT        NOT NULL DEFAULT 'RUNNING',
    state           JSONB       NOT NULL DEFAULT '{}',
    budget_used     NUMERIC(10,6) NOT NULL DEFAULT 0,
    budget_cap      NUMERIC(10,6) NOT NULL DEFAULT 5.0,
    current_step    INTEGER     NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_states_task_id   ON agent_states (task_id);
CREATE INDEX IF NOT EXISTS idx_agent_states_status    ON agent_states (status);
CREATE INDEX IF NOT EXISTS idx_agent_states_provider  ON agent_states (provider_id);

-- ─── eval_traces ────────────────────────────────────────────────────────────
-- Tiered eval storage (§1.5).
-- storage_tier: 'HOT' | 'WARM' | 'COLD'
CREATE TABLE IF NOT EXISTS eval_traces (
    trace_id        TEXT        PRIMARY KEY,
    eval_id         TEXT        NOT NULL,
    scenario_id     TEXT        NOT NULL,
    provider_id     TEXT,
    task_type       TEXT        NOT NULL DEFAULT 'generic',
    final_verdict   TEXT        NOT NULL DEFAULT 'FAILURE',
    quality_score   SMALLINT    NOT NULL DEFAULT 0 CHECK (quality_score BETWEEN 0 AND 100),
    safety_score    SMALLINT    NOT NULL DEFAULT 100 CHECK (safety_score BETWEEN 0 AND 100),
    latency_ms      INTEGER     NOT NULL DEFAULT 0,
    cost_incurred   NUMERIC(12,6) NOT NULL DEFAULT 0,
    storage_tier    TEXT        NOT NULL DEFAULT 'HOT',
    allow_training  BOOLEAN     NOT NULL DEFAULT FALSE,
    -- HOT: full payload; WARM: no raw_output; COLD: scrubbed
    eval_input      JSONB       NOT NULL DEFAULT '{}',
    eval_result     JSONB       NOT NULL DEFAULT '{}',
    raw_output      JSONB       NOT NULL DEFAULT '{}',  -- cleared on WARM promotion
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ            -- set when tier is upgraded
);

CREATE INDEX IF NOT EXISTS idx_eval_traces_scenario     ON eval_traces (scenario_id);
CREATE INDEX IF NOT EXISTS idx_eval_traces_provider     ON eval_traces (provider_id);
CREATE INDEX IF NOT EXISTS idx_eval_traces_tier         ON eval_traces (storage_tier);
CREATE INDEX IF NOT EXISTS idx_eval_traces_verdict      ON eval_traces (final_verdict);
CREATE INDEX IF NOT EXISTS idx_eval_traces_eval_input   ON eval_traces USING GIN (eval_input);
CREATE INDEX IF NOT EXISTS idx_eval_traces_eval_result  ON eval_traces USING GIN (eval_result);

-- ─── execution_traces ───────────────────────────────────────────────────────
-- Lightweight per-step execution log (§5.2 Execution Trace).
CREATE TABLE IF NOT EXISTS execution_traces (
    run_id               TEXT        PRIMARY KEY,
    task_id              TEXT        NOT NULL,
    agent_state_id       TEXT        REFERENCES agent_states(agent_state_id),
    provider_id          TEXT,
    request_id           TEXT        NOT NULL,
    latency_ms           INTEGER     NOT NULL DEFAULT 0,
    tool_calls_count     INTEGER     NOT NULL DEFAULT 0,
    context_tokens_used  INTEGER     NOT NULL DEFAULT 0,
    budget_used          NUMERIC(10,6) NOT NULL DEFAULT 0,
    budget_cap           NUMERIC(10,6) NOT NULL DEFAULT 5.0,
    error                JSONB       NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_traces_task_id  ON execution_traces (task_id);
CREATE INDEX IF NOT EXISTS idx_execution_traces_provider ON execution_traces (provider_id);

-- ─── routing_decisions ──────────────────────────────────────────────────────
-- Minimal routing log (Phase 1 stub; full Cost Router in Phase 2).
CREATE TABLE IF NOT EXISTS routing_decisions (
    decision_id          TEXT        PRIMARY KEY,
    task_id              TEXT        NOT NULL,
    chosen_provider_id   TEXT        NOT NULL,
    fallback_provider_id TEXT,
    mode                 TEXT        NOT NULL DEFAULT 'conservative',
    reason               TEXT        NOT NULL DEFAULT '',
    payload              JSONB       NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Seed: Claude provider profile ─────────────────────────────────────────
INSERT INTO provider_profiles (provider_id, profile)
VALUES (
    'claude_3_5_sonnet',
    '{
        "provider_id": "claude_3_5_sonnet",
        "capabilities": {
            "max_context_tokens": 200000,
            "supports_tools": true,
            "supports_images": true,
            "supports_parallel_agents": false,
            "max_parallel_agents": 1,
            "supports_arena_mode": false,
            "supports_context_circulation": false,
            "supports_3_agent_harness": false,
            "native_multi_turn": true,
            "streaming": true,
            "supports_streaming_cost_tracking": false,
            "supports_budget_interrupt": true
        },
        "pricing": { "input_per_1m": 3.0, "output_per_1m": 15.0 },
        "known_strengths": ["reasoning", "code", "eval", "safety"],
        "known_weaknesses": ["very_long_autonomy"],
        "policies": {
            "can_route_to_competitors": true,
            "can_store_outputs_for_training": false,
            "can_use_in_eval_comparison": true,
            "data_residency_requirements": ["global"],
            "prohibited_use_cases": []
        }
    }'::jsonb
)
ON CONFLICT (provider_id) DO NOTHING;

COMMIT;
