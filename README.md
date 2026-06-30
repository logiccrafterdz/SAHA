# SAHA – Standardized Agent Harness Architecture

**Core Implementation** | Python 3.12 + Go 1.22 | Redis + PostgreSQL

This README is written to be consumable by both human engineers and AI coding agents. It follows SAHA‑Spec contracts and can be referenced in `AGENTS.md` or `HARNESS.md` for agent workflows.

---

## Overview

For full architectural contracts, see [SAHA-Spec.md](./SAHA-Spec.md).

SAHA separates AI agent orchestration into clean, contract-driven layers:

| Layer | Language | Port | Responsibility |
|-------|----------|------|----------------|
| **Execution Harness** | Python / FastAPI | 8001 | Multi-turn agent loop, budget tracking |
| **Vendor Abstraction** | Python / FastAPI | 8002 | Unified provider interface, PrivacyGate |
| **Neutral Eval Harness** | Python / FastAPI | 8003 | Provider-agnostic grading, LLM-as-Judge |
| **Observability API** | Python / FastAPI | 8004 | Metrics aggregation, anomaly detection |
| **Routing API** | Python / FastAPI | 8005 | Cost routing, escalation policies, HITL |
| **Event Bus (Go)** | Go | 8090 | Redis pub/sub broker, topic routing |
| **PostgreSQL** | — | 5432 | agent_state, eval_traces, metrics, hitl |
| **Redis** | — | 6379 | Async message bus |

---

## Use Cases

| Use Case | What SAHA does | Relevant APIs |
|----------|----------------|---------------|
| **Multi‑provider code‑fix** | Route tasks between Claude/GPT/Gemini, evaluate, log | `:8001`, `:8003`, `:8005` |
| **Offline eval of outputs** | Grade arbitrary outputs with Eval Harness | `:8003` |
| **Monitoring provider performance** | Aggregated quality/cost/latency metrics | `:8004` |

> **Note:** For routing and observability to work effectively, define `scenario_id` and `success_contract` consistently across tasks. SAHA will record `routing_decisions`, `eval_traces`, and `provider_stats` automatically.

---

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and DB credentials
```

### 2. Run with Docker Compose

> **Note:** For multi‑provider routing, configure `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY` in `.env` and corresponding `profiles/*.json`.

```bash
docker compose up --build
```

Services start in dependency order:
`postgres + redis → saha-bus → execution-api + vendor-api + eval-api + observability-api + routing-api`

### 3. Verify all services are healthy

```bash
curl http://localhost:8001/health   # execution-api
curl http://localhost:8002/health   # vendor-api
curl http://localhost:8003/health   # eval-api
curl http://localhost:8004/health   # observability-api
curl http://localhost:8005/health   # routing-api
curl http://localhost:8090/health   # saha-bus (Go)
```

---

## API Reference

### Execution API `:8001`

#### `POST /tasks/run`
Run the full agent loop for a task.

```json
// Note: Omit `provider_id` to enable dynamic Cost Routing, or specify any configured model.
{
  "task_id": "optional-uuid",
  "provider_id": "claude_3_5_sonnet",
  "message": "Write a Python function that sorts a list",
  "system_prompt": "You are a senior Python engineer.",
  "scenario_id": "CODE_GEN_PY",
  "domain_tags": ["python", "code"],
  "success_contract": {
    "must_pass_tests": false,
    "max_tool_calls": 10,
    "allowed_error_types": ["NONE"]
  },
  "options": {
    "budget_cap": 2.00,
    "routing_mode": "conservative"
  }
}
```

**Response:**
```json
{
  "agent_state_id": "uuid",
  "task_id": "uuid",
  "status": "COMPLETED",
  "budget_used": 0.0042,
  "current_step": 0
}
```

#### `GET /tasks/{agent_state_id}`
Retrieve the full AgentState for a completed task.

---

### Vendor API `:8002`

#### `GET /providers`
List all registered provider IDs.

#### `GET /providers/{provider_id}/profile`
Get capability & policy profile for a provider.

#### `POST /complete?provider_id=claude_3_5_sonnet`
Send a UnifiedAgentRequest directly to a provider (bypasses agent loop).

---

### Eval API `:8003`

#### `POST /eval`
Synchronously grade an EvalInput payload.

```json
{
  "task_type": "code_generation",
  "scenario_id": "CODE_GEN_PY",
  "normalized_output": { "text": "def sort_list(lst): return sorted(lst)" },
  "success_contract": { "max_tool_calls": 5 },
  "context": { "tool_calls_count": 1 }
}
```

**Response:**
```json
{
  "final_verdict": "SUCCESS",
  "quality_score": 95,
  "safety_score": 100,
  "grader_confidence": 100,
  "grader_breakdown": { ... }
}
```

#### `POST /traces/{trace_id}/promote-warm`
Promote HOT trace → WARM (drops raw_output).

#### `POST /traces/{trace_id}/promote-cold`
Promote WARM trace → COLD (PII scrubbed).

---

### Observability API `:8004`

#### `GET /metrics/providers`
Get aggregated performance/cost stats.

#### `GET /anomalies`
Get active anomalies (QUALITY_DROP, COST_SPIKE).

---

### Routing API `:8005`

#### `POST /route/decide`
Get a routing decision for a TaskProfile.

#### `POST /hitl/override`
Apply real-time constraint overrides (§6.2.1).

---

### Event Bus (Go) `:8090`

#### `GET /health`
Returns bus health + Redis connectivity status.

#### `GET /topics`
Lists all active SAHA topics.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     SAHA Architecture                     │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │     Go Event Bus (saha-bus) – Redis pub/sub      │    │
│  │  Topics: SAHA/agent_requests, SAHA/eval_inputs,  │    │
│  │          SAHA/anomaly_alerts, etc.               │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                   │
│  ┌────────────────────┼────────────────────────────┐    │
│  │    Python Services (FastAPI + asyncio)           │    │
│  │                                                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │Execution │  │ Vendor   │  │   Eval   │       │    │
│  │  │(Loop)    │◄►│ +Privacy │  │ +LLMJudge│       │    │
│  │  └────┬─────┘  └──────────┘  └──────────┘       │    │
│  │       │                                         │    │
│  │  ┌────▼─────┐  ┌──────────┐                     │    │
│  │  │ Routing  │◄─│ Observab.│                     │    │
│  │  │ +HITL    │  │ +Anomaly │                     │    │
│  │  └──────────┘  └──────────┘                     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │    PostgreSQL 16 (JSONB)                         │    │
│  │    agent_states, eval_traces, provider_stats,    │    │
│  │    routing_decisions, hitl_overrides             │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Development

### Install Python dependencies

```bash
pip install -e ".[dev]"
```

### Run unit tests (no DB/Redis required)

```bash
pytest tests/unit/ -v
```

### Run integration tests (no external services)

```bash
pytest tests/integration/ -v
```

### Run all tests with coverage

```bash
pytest --cov=saha --cov-report=html
```

### Lint and type-check

```bash
ruff check saha/ tests/
mypy saha/
```

### Build the Go event bus locally

```bash
cd bus
go mod tidy
go build -o saha-bus ./main.go
./saha-bus
```

---

## Project Structure

```
SAHA/
├── bus/                        # Go Event Bus
│   ├── main.go                 # Entry point (HTTP :8090 + Redis)
│   ├── internal/bus/           # Core bus logic + topic constants
│   └── internal/redis/         # Redis pub/sub adapter
│
├── saha/                       # Python core package
│   ├── contracts/              # Pydantic v2 data contracts (SAHA spec)
│   ├── vendor/                 # Vendor Abstraction Layer
│   │   ├── base.py             # BaseAdapter ABC
│   │   ├── error_mapper.py     # Error taxonomy mapping
│   │   ├── adapters/           # Claude, Gemini, OpenAI adapters
│   │   └── profiles/           # Capability and tier definitions
│   ├── execution/              # Agent Execution Harness
│   │   ├── agent_loop.py       # Multi-turn loop (§3.3)
│   │   ├── agent_state.py      # DB-backed state manager
│   │   └── tool_runner.py      # Tool execution sandbox
│   ├── eval/                   # Neutral Eval Harness
│   │   ├── normalizer.py       # Normalization pipeline (§1.4)
│   │   ├── grader.py           # Deterministic grader + LLM stub
│   │   └── storage.py          # Hot/Warm/Cold trace storage
│   ├── routing/                # Cost Router & Escalations
│   ├── observability/          # Metrics Aggregator & Anomalies
│   ├── privacy/                # Data Residency & PII Redaction
│   ├── hitl/                   # Human-in-the-loop Controls
│   ├── event_bus/              # Redis bus client (Python)
│   └── db/                     # asyncpg pool + migrations
│
├── services/                   # FastAPI entry points
│   ├── execution_api/main.py       # :8001
│   ├── vendor_api/main.py          # :8002
│   ├── eval_api/main.py            # :8003
│   ├── observability_api/main.py   # :8004
│   └── routing_api/main.py         # :8005
│
├── tests/
│   ├── unit/                   # No external deps required
│   └── integration/            # Mocked bus + state manager
│
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Core Architecture Highlights

| Feature | Implementation | Description |
|---------|----------------|-------------|
| **Cost Routing System** | `saha/routing/router.py` | Automated provider selection based on eval stats |
| **Observability** | `saha/observability/` | Quality/cost/latency reports via Metrics Aggregator |
| **LLM-as-Judge** | `saha/eval/llm_judge.py` | Full grader with Claude judge + Calibration |
| **Provider Adapters** | `saha/vendor/adapters/` | Claude 3.x, GPT-4.x, Gemini 1.x (configurable via env/profiles) |
| **HITL Controls API** | `saha/hitl/service.py` | Policy override, judge calibration, failure triage endpoints |
| **Privacy Gate** | `saha/privacy/gate.py` | PII redaction, Data Residency validation, Output Scanning |

*Optional Enhancements: The Redis pub/sub adapter in `bus/internal/` can be swapped for Kafka/RabbitMQ if desired.*

---

## Spec Compliance

All contracts implement [SAHA-Spec.md](./SAHA-Spec.md):

| Spec Section | Implementation |
|---|---|
| §1.2 Eval Input | `saha/contracts/eval.py → EvalInput` |
| §1.3 Eval Result | `saha/contracts/eval.py → EvalResult` |
| §1.4 Normalization | `saha/eval/normalizer.py` |
| §1.5 Trace Storage | `saha/eval/storage.py` |
| §2.2 Unified Request | `saha/contracts/vendor.py → UnifiedAgentRequest` |
| §2.3 Unified Response | `saha/contracts/vendor.py → UnifiedAgentResponse` |
| §2.4 Provider Profile | `saha/contracts/vendor.py → ProviderProfile` |
| §2.5 ErrorMapper | `saha/vendor/error_mapper.py` |
| §3.2 Agent State | `saha/contracts/execution.py → AgentState` |
| §3.3 Execution Loop | `saha/execution/agent_loop.py` |
| §3.4 Budget Interrupt | `saha/contracts/vendor.py → BudgetInterruptSignal` |
| §4.2 Routing Decision | `saha/routing/router.py` |
| §5.3 Privacy Gate | `saha/privacy/gate.py` |

---

## End-to-End Scenario: `SCENARIO_PY_FIX`

This is the reference scenario for SAHA. Use it to verify the full
pipeline is working after standing up the stack with `docker compose up`.

### 1 — Send the Task

```bash
curl -s -X POST http://localhost:8001/tasks/run \
  -H "Content-Type: application/json" \
  -d '{
    // Note: Omit provider_id to enable dynamic Cost Routing
    "provider_id":   "claude_3_5_sonnet",
    "message":       "Fix the divide-by-zero bug in calculate_average(nums). Return only the corrected function.",
    "system_prompt": "You are a senior Python engineer. Respond with corrected Python code only.",
    "scenario_id":   "SCENARIO_PY_FIX",
    "domain_tags":   ["python", "code"],
    "success_contract": {
      "must_pass_tests":     false,
      "must_pass_linter":    false,
      "max_tool_calls":      5,
      "allowed_error_types": ["NONE"]
    },
    "options": {
      "budget_cap":    1.00,
      "routing_mode":  "conservative"
    }
  }'
```

**Expected Response:**

```json
{
  "agent_state_id": "3f8a1b2c-...",
  "task_id":        "9d4e7f1a-...",
  "status":         "COMPLETED",
  "budget_used":    0.0042,
  "current_step":   0
}
```

---

### 2 — Inspect DB State

#### `agent_states` — task runtime record
```sql
SELECT agent_state_id, task_id, provider_id, status,
       budget_used, current_step, updated_at
FROM   agent_states
WHERE  status = 'COMPLETED'
ORDER  BY updated_at DESC
LIMIT  1;
```
```
 agent_state_id | task_id | provider_id        | status    | budget_used | current_step
----------------+---------+--------------------+-----------+-------------+-------------
 3f8a1b2c-...   | 9d4e... | claude_3_5_sonnet  | COMPLETED |   0.004200  |     0
```

#### `routing_decisions` — CostRouter decision log

*(Note: In this specific scenario we provided `provider_id` explicitly. If omitted, the CostRouter selects the optimal provider dynamically).*
```sql
SELECT decision_id, chosen_provider_id, mode, reason, created_at
FROM   routing_decisions
ORDER  BY created_at DESC
LIMIT  1;
```
```
 decision_id | chosen_provider_id | mode          | reason
 a1b2c3d4-.. | claude_3_5_sonnet  | conservative  | {"quality_score": 95.0, "safety_score": 100, "rank": 1}
```

---

### 3 — Eval Trace (via Eval API)

The Eval Harness processes the result asynchronously via the event bus.
Check the result directly:

```bash
curl -s -X POST http://localhost:8003/eval \
  -H "Content-Type: application/json" \
  -d '{
    "task_type":   "code_generation",
    "scenario_id": "SCENARIO_PY_FIX",
    "normalized_output": {
      "text": "def calculate_average(nums):\n    if not nums:\n        return 0\n    return sum(nums) / len(nums)"
    },
    "success_contract": {
      "max_tool_calls":      5,
      "allowed_error_types": ["NONE"]
    },
    "context": { "tool_calls_count": 0, "context_tokens_used": 470 }
  }'
```

#### EvalTrace — SUCCESS
```json
{
  "final_verdict":     "SUCCESS",
  "quality_score":     100,
  "safety_score":      100,
  "grader_confidence": 100,
  "latency_ms":        12,
  "grader_breakdown": {
    "deterministic_checks": [
      { "check": "max_tool_calls",          "passed": true,  "expected": "<= 5", "actual": 0 },
      { "check": "allowed_error_types",     "passed": true,  "expected": ["NONE"], "actual": "NONE" },
      { "check": "no_refusal",              "passed": true },
      { "check": "no_hallucination_signals","passed": true },
      { "check": "non_empty_output",        "passed": true }
    ],
    "llm_judge":  { "enabled": true, "score": 100, "note": "Properly handled empty list edge case." },
    "human_review": {}
  }
}
```

#### EvalTrace — FAILURE (refusal example)
```json
{
  "final_verdict":     "FAILURE",
  "quality_score":     60,
  "safety_score":      70,
  "grader_confidence": 40,
  "grader_breakdown": {
    "deterministic_checks": [
      { "check": "max_tool_calls",      "passed": true },
      { "check": "allowed_error_types", "passed": true },
      { "check": "no_refusal",          "passed": false, "detail": "Refusal language detected in output" },
      { "check": "no_hallucination_signals", "passed": true },
      { "check": "non_empty_output",    "passed": true }
    ]
  }
}
```

#### `eval_traces` — tiered storage in PostgreSQL
```sql
SELECT trace_id, scenario_id, final_verdict, quality_score,
       safety_score, storage_tier, created_at
FROM   eval_traces
WHERE  scenario_id = 'SCENARIO_PY_FIX'
ORDER  BY created_at DESC;
```
```
 trace_id     | scenario_id      | final_verdict | quality_score | safety_score | storage_tier
 7a3b9c1d-... | SCENARIO_PY_FIX  | SUCCESS       |           100 |          100 | HOT
```

---

### 4 — Promote Trace to WARM (drop raw_output)

```bash
curl -s -X POST http://localhost:8003/traces/7a3b9c1d-.../promote-warm
# → {"trace_id": "7a3b9c1d-...", "new_tier": "WARM"}
```

After promotion, `raw_output = {}` in the DB, `normalized_output` and all
metrics are preserved for routing and trend analysis.

---

### 5 — Bus Activity (Go bus logs)

While running the scenario, the Go event bus at `:8090` logs:

```
{"level":"INFO","topic":"SAHA/eval_inputs",      "keys":["eval_id","scenario_id","provider_info",...]}
{"level":"INFO","topic":"SAHA/eval_results",     "keys":["eval_id","final_verdict","quality_score",...]}
```

Check health and active topics:
```bash
curl http://localhost:8090/health
curl http://localhost:8090/topics
```
