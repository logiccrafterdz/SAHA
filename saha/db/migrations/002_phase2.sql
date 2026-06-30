-- SAHA Phase 2 – DB Schema Extensions
-- Migration 002: Observability, HITL, Calibration tables
-- Spec ref: §4.2, §5.2, §6.2

-- Provider aggregated stats (consumed by Cost Router §4.3)
CREATE TABLE IF NOT EXISTS provider_stats (
    provider_id     TEXT        NOT NULL,
    scenario_id     TEXT        NOT NULL,
    window          TEXT        NOT NULL,   -- '24h' | '7d' | '30d'
    quality_p50     REAL        DEFAULT 0,
    quality_p90     REAL        DEFAULT 0,
    safety_avg      REAL        DEFAULT 0,
    success_rate    REAL        DEFAULT 0,
    error_rate      REAL        DEFAULT 0,
    cost_per_task   REAL        DEFAULT 0,
    latency_p50_ms  INTEGER     DEFAULT 0,
    latency_p90_ms  INTEGER     DEFAULT 0,
    sample_count    INTEGER     DEFAULT 0,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (provider_id, scenario_id, window)
);
CREATE INDEX IF NOT EXISTS provider_stats_window_idx ON provider_stats (window, computed_at);

-- HITL router policy overrides (§6.2.1)
CREATE TABLE IF NOT EXISTS hitl_overrides (
    override_id   TEXT        PRIMARY KEY,
    scope         TEXT        NOT NULL,    -- 'project_X' | 'global'
    change        JSONB       NOT NULL,    -- {quality_min, routing_mode, default_provider_id, ...}
    reason        TEXT        NOT NULL,
    approved_by   TEXT        NOT NULL,
    active        BOOLEAN     DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Success contract version history (§6.2.3)
CREATE TABLE IF NOT EXISTS success_contract_history (
    contract_id   TEXT        PRIMARY KEY,
    scenario_id   TEXT        NOT NULL,
    old_contract  JSONB       NOT NULL,
    new_contract  JSONB       NOT NULL,
    reason        TEXT        NOT NULL,
    approved_by   TEXT        NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS sc_history_scenario_idx ON success_contract_history (scenario_id);

-- Judge calibration runs (§6.2.2)
CREATE TABLE IF NOT EXISTS judge_calibration_runs (
    run_id              TEXT    PRIMARY KEY,
    golden_dataset_size INTEGER DEFAULT 0,
    cases_run           INTEGER DEFAULT 0,
    mean_quality_dev    REAL    DEFAULT 0,
    mean_safety_dev     REAL    DEFAULT 0,
    max_deviation       REAL    DEFAULT 0,
    judge_enabled       BOOLEAN DEFAULT TRUE,
    recommendation      TEXT    DEFAULT '',
    report              JSONB   DEFAULT '{}',
    duration_ms         INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Anomaly detection log (§5.1)
CREATE TABLE IF NOT EXISTS anomaly_log (
    anomaly_id  TEXT        PRIMARY KEY,
    type        TEXT        NOT NULL,   -- 'QUALITY_DROP'|'COST_SPIKE'|'THRASHING'|'SAFETY_VIOLATION'
    provider_id TEXT,
    scenario_id TEXT,
    severity    TEXT        DEFAULT 'WARNING',
    details     JSONB       DEFAULT '{}',
    resolved    BOOLEAN     DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS anomaly_log_type_idx ON anomaly_log (type, created_at);

-- Failure triage records (§6.2.4)
CREATE TABLE IF NOT EXISTS failure_triages (
    incident_id        TEXT        PRIMARY KEY,
    run_id             TEXT,
    eval_id            TEXT,
    classified_root    TEXT        NOT NULL,
    notes              TEXT        DEFAULT '',
    action_items       JSONB       DEFAULT '[]',
    resolved           BOOLEAN     DEFAULT FALSE,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
