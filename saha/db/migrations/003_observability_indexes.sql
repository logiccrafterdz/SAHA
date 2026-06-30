-- SAHA Phase 2 - Migration 003: Observability Performance Indexes
-- Ensures fast querying for Observability API over large datasets.

BEGIN;

-- For fast aggregation of completion stats
CREATE INDEX IF NOT EXISTS idx_agent_states_status_updated
    ON agent_states (status, updated_at);

-- For fast time-series queries by scenario
CREATE INDEX IF NOT EXISTS idx_eval_traces_scenario_created
    ON eval_traces (scenario_id, created_at);

-- For fast archival / tier management queries
CREATE INDEX IF NOT EXISTS idx_eval_traces_tier_created
    ON eval_traces (storage_tier, created_at);

COMMIT;
