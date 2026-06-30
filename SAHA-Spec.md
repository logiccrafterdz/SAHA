
# SAHA – Standardized Agent Harness Architecture  
## Layered Design & Contracts (Updated)

### 0. Purpose, Scope, and Execution Model

SAHA is a standardized architecture for AI agent harnesses. It cleanly separates:

- **Neutral Evaluation Layer** (Neutral Eval Harness)  
- **Provider Abstraction Layer** (Vendor Abstraction)  
- **Agent Execution Layer** (Execution Harness / Agent Loop)  
- **Cost & Quality Routing Layer** (Cost Routing System)  
- **Observability & Privacy Layer** (Observability & Privacy Gate)  
- **Human‑in‑the‑Loop Controls** (HITL)

**Goal:**  
Run multiple models (Claude, Grok, Qwen, Kimi, GPT, Gemini, etc.) under the same harness, evaluate them neutrally, orchestrate multi‑step agent loops, and control cost, quality, and safety in production.

**Execution Model:**  
The contracts defined in SAHA describe **payload structures**, not necessarily synchronous HTTP calls. In production, SAHA is intended to run over an **event‑driven infrastructure** (e.g., Kafka, RabbitMQ, internal queues):

- Requests, responses, routing decisions, eval results, and traces are **messages on a bus**.  
- Components subscribe to topics (e.g., `agent_requests`, `provider_responses`, `eval_results`) and act asynchronously.  
- Synchronous semantics (request/response) may be emulated where needed, but the architecture assumes **non‑blocking, event‑driven flows** for long‑running and multi‑agent tasks.

***

## 1. Neutral Eval Harness

### 1.1 Responsibility

The Neutral Eval Harness is responsible for:

- Defining **Success Contracts** per scenario.  
- Running evaluations on any agent output, independent of provider identity.  
- Producing metrics for routing and improvement.  
- Remaining **agnostic to the provider** during grading (except for logging).

### 1.2 Input Contract (Eval Input)

Each evaluation receives a unified input payload:

```json
{
  "eval_id": "UUID",
  "task_type": "code_generation | data_extraction | semantic_search | ...",
  "scenario_id": "SCENARIO_123",
  "domain_tags": ["python", "finance", "internal"],
  "input_normalized": { ... },        // canonical request
  "normalized_output": { ... },       // provider output after normalization
  "success_contract": {
    "must_pass_tests": true,
    "must_pass_linter": false,
    "max_tool_calls": 10,
    "allowed_error_types": ["NONE", "MODEL_ERROR.REFUSAL.INFO"],
    "custom_rubric": "Rate explanation clarity from 1 to 5"
  },
  "provider_info": {
    "provider_id": "qwen_3.7_max",
    "run_id": "RUN_UUID",
    "raw_output_ref": "storage://hot/..."
  },
  "context": {
    "tool_calls_count": 7,
    "context_tokens_used": 8500
  }
}
```

### 1.3 Output Contract (Eval Result)

The Eval Harness returns:

```json
{
  "eval_id": "UUID",
  "scenario_id": "SCENARIO_123",
  "final_verdict": "SUCCESS | FAILURE | PARTIAL",
  "quality_score": 0-100,
  "safety_score": 0-100,
  "latency_ms": 1234,
  "cost_incurred": 0.37,
  "tool_calls_count": 7,
  "context_tokens_used": 8500,
  "error_type": "NONE | MODEL_ERROR.HALLUCINATION.CRITICAL | ...",
  "grader_confidence": 0-100,
  "grader_breakdown": {
    "deterministic_checks": [...],
    "llm_judge": {...},
    "human_review": {...}
  }
}
```

### 1.4 Normalization Pipeline Contract

Before any grading, outputs pass through a normalization pipeline:

**Input:** `raw_output`, `provider_id`  
**Output:** `normalized_output`

Rules:

- Strip provider‑specific metadata (internal IDs, headers, extra wrapper fields).  
- Canonicalize JSON (keys, ordering irrelevant but structure normalized).  
- Extract the core payload relevant to the scenario and success_contract.  
- Any failure in normalization is recorded as `EVAL_ERROR.GRADER_FAILURE.CRITICAL`.

### 1.5 Eval Trace Storage Contract

Each eval run produces an **Eval Trace** stored in tiers:

1. **Hot Storage (≤ 30 days)**  
   - Full trace: `raw_output`, `normalized_output`, success_contract, grader decisions, metrics.  
   - Used for incident debugging and immediate analysis.

2. **Warm Storage (≈ 3–6 months)**  
   - Drops `raw_output`, keeps `normalized_output` + metrics + contracts.  
   - Used for cost routing, drift detection, trend analysis.

3. **Cold/Training Storage (long‑term)**  
   - PII scrubbed; keeps `task_type`, `scenario_id`, `success_contract`, `final_verdict` + selected metrics.  
   - Serves as dataset for improving models, tools, and judges (subject to training flags).

The Privacy Gate mediates all transitions between Hot/Warm/Cold tiers.

***

## 2. Vendor Abstraction Layer

### 2.1 Responsibility

The Vendor Abstraction Layer:

- Provides a unified interface for agent requests/responses.  
- Translates unified requests into provider‑specific formats (Adapters).  
- Maps provider responses and errors into canonical forms.  
- Maintains Provider Capability & Policy Profiles.

### 2.2 Unified Agent Request Contract

Upper layers (Execution Harness, Cost Router) send requests in this canonical form:

```json
{
  "request_id": "UUID",
  "task_id": "TASK_UUID",
  "agent_state_id": "AGENT_STATE_UUID",
  "message": "...",
  "system_prompt": "...",
  "tools": [
    { "name": "read_file", "schema": {...} },
    { "name": "run_tests", "schema": {...} }
  ],
  "options": {
    "max_context_tokens": 1000000,
    "enable_arena_mode": true,
    "enable_context_circulation": true,
    "harness_pattern": "3_agent | single_agent | swarm",
    "parallel_agents": 8,
    "budget_cap": 5.00,
    "routing_mode": "conservative | exploratory"
  }
}
```

Required fields must be supported by all providers (or emulated); optional fields are best‑effort and ignored if unsupported.

### 2.3 Unified Agent Response Contract

Each provider adapter returns:

```json
{
  "request_id": "UUID",
  "provider_id": "grok_build_cli",
  "run_id": "RUN_UUID",
  "status": "COMPLETED | NEEDS_TOOL | FAILED",
  "normalized_output": { ... },          // after light normalization
  "raw_output_ref": "storage://hot/...", // reference to raw result
  "tool_calls_count": 7,
  "context_tokens_used": 8500,
  "cost_estimate": 0.25,
  "latency_ms": 800,
  "error": {
    "type": "NONE | MODEL_ERROR | TOOL_ERROR | INFRA_ERROR | POLICY_ERROR | EVAL_ERROR",
    "code": "HALLUCINATION | PROVIDER_RATE_LIMIT | ...",
    "severity": "INFO | WARNING | CRITICAL",
    "details": "..."
  }
}
```

For long‑running tasks, `cost_estimate` may be partial and updated over time via streaming events.

### 2.4 Provider Capability & Policy Profile Contract

Each provider has a profile:

```json
{
  "provider_id": "qwen_3.7_max",
  "capabilities": {
    "max_context_tokens": 1000000,
    "supports_tools": true,
    "supports_images": true,
    "supports_parallel_agents": true,
    "max_parallel_agents": 100,
    "supports_arena_mode": false,
    "supports_context_circulation": false,
    "supports_3_agent_harness": false,
    "native_multi_turn": true,
    "streaming": true,
    "supports_streaming_cost_tracking": true,
    "supports_budget_interrupt": true
  },
  "pricing": {
    "input_per_1m": 1.25,
    "output_per_1m": 3.75
  },
  "known_strengths": ["long_autonomy", "code", "multilingual"],
  "known_weaknesses": ["subtle_reasoning", "safety_compliance"],
  "policies": {
    "can_route_to_competitors": true,
    "can_store_outputs_for_training": false,
    "can_use_in_eval_comparison": true,
    "data_residency_requirements": ["CN", "global"],
    "prohibited_use_cases": ["medical_diagnosis", "legal_advice"]
  }
}
```

Profiles are updated periodically based on observed performance (via Observability), subject to minimum sample size to avoid overfitting to small numbers of tasks.

### 2.5 ErrorMapper Contract

The `ErrorMapper` component:

- Accepts raw error signals (HTTP codes, SDK exceptions, CLI output).  
- Maps them into the canonical taxonomy:

  - `type`: MODEL_ERROR / TOOL_ERROR / INFRA_ERROR / POLICY_ERROR / EVAL_ERROR  
  - `code`: specific error (HALLUCINATION, TOOL_TIMEOUT, PROVIDER_RATE_LIMIT, etc.)  
  - `severity`: INFO / WARNING / CRITICAL

- Unknown errors map to `INFRA_ERROR.UNKNOWN.WARNING` with details preserved.

***

## 3. Agent Execution Harness (Agent Loop)

### 3.1 Responsibility

The Execution Harness is the “agent runtime”:

- Orchestrates the **agent execution loop** (multi‑turn behavior).  
- Maintains **agent state** (memory, progress, budget).  
- Interprets `status: NEEDS_TOOL` from Vendor Abstraction.  
- Executes tools and feeds their results back to the provider.  
- Decides when a task is complete and when to hand output to the Neutral Eval Harness.  
- Uses Cost Router mainly at **task start** and during **escalation**, not on every micro‑step.

### 3.2 Agent State Contract

Each agent/task has a state record:

```json
{
  "agent_state_id": "UUID",
  "task_id": "TASK_UUID",
  "provider_id": "kimi_k2.5",
  "current_step": 12,
  "status": "RUNNING | WAITING_FOR_TOOL | COMPLETED | FAILED",
  "memory": {
    "short_term": { ... },    // recent messages, decisions
    "long_term_ref": "storage://memory/agent_state_id"
  },
  "pending_tool_call": {
    "name": "read_file",
    "arguments": { "path": "..." }
  },
  "budget_used": 3.10,
  "budget_cap": 5.00,
  "context_tokens_used": 480000
}
```

### 3.3 Agent Execution Loop (Behavior Contract)

High‑level behavior:

1. **Start:**  
   - Cost Router selects `provider_id` for `task_id`.  
   - Execution Harness initializes `agent_state` (step = 0, status = RUNNING, memory empty, budget_used = 0).

2. **Loop step:**

   - Construct `Unified Agent Request` from `agent_state` and task description.  
   - Send to Vendor Abstraction with chosen `provider_id`.  
   - Receive unified response:

     - `status = COMPLETED`  
       - Set `agent_state.status = COMPLETED`.  
       - Compose `Eval Input` (including `success_contract`, normalized output, context) and send to Neutral Eval Harness.

     - `status = NEEDS_TOOL`  
       - Read `pending_tool_call` from response.  
       - Execute tool locally or via tool harness.  
       - Update `memory.short_term` with tool results.  
       - Increment `current_step`.  
       - Repeat loop with same `provider_id` (no new routing decision).

     - `status = FAILED`  
       - Mark failure; may trigger escalation via Cost Router or HITL intervention.

3. **Budget Check per step:**

   - If provider supports streaming cost tracking, Execution Harness updates `budget_used` with partial `cost_estimate`.  
   - If `budget_used` is about to exceed `budget_cap`, emit **Budget Interrupt Signal** to Vendor Abstraction.

### 3.4 Budget Interrupt Signal Contract

When budget is exceeded or critically close:

```json
{
  "command_id": "UUID",
  "task_id": "TASK_UUID",
  "run_id": "RUN_UUID",
  "provider_id": "kimi_k2.5",
  "reason": "BUDGET_CAP_REACHED",
  "budget_cap": 5.00,
  "budget_used": 5.02
}
```

Vendor Abstraction:

- Attempts to cancel/stop the long‑running job via provider’s mechanisms.  
- Returns a unified response with:

```json
"error": {
  "type": "POLICY_ERROR",
  "code": "BUDGET_EXCEEDED",
  "severity": "CRITICAL",
  "details": "Job stopped due to budget cap 5.00"
}
```

***

## 4. Cost Routing System

### 4.1 Responsibility

The Cost Routing System:

- Selects providers per task based on:  
  - Provider capabilities & policies.  
  - Historical eval stats (quality, safety, cost).  
  - Task profile and routing mode.  
- Applies escalation and stability policies.  
- Enforces hard constraints on quality, safety, latency, and error types.

### 4.2 Routing Decision Contract

Input:

```json
{
  "task_profile": {
    "task_id": "TASK_UUID",
    "task_type": "code_generation",
    "scenario_id": "SCENARIO_PY_FIX",
    "domain_tags": ["python", "internal"],
    "importance": "CRITICAL | NORMAL | LOW",
    "budget_cap": 2.00,
    "routing_mode": "conservative | exploratory"
  },
  "candidate_providers": ["claude_opus", "kimi_k2.5", "qwen_3.7_max"],
  "provider_profiles": { ... },    // as per section 2.4
  "recent_eval_stats": {
    "claude_opus": {...},
    "kimi_k2.5": {...},
    "qwen_3.7_max": {...}
  }
}
```

Output:

```json
{
  "decision_id": "UUID",
  "chosen_provider_id": "kimi_k2.5",
  "fallback_provider_id": "claude_opus",
  "constraints_applied": {
    "quality_min": 75,
    "safety_min": 90,
    "latency_max_ms": 30000,
    "error_types_forbidden": ["HALLUCINATION", "SAFETY_POLICY_VIOLATION"]
  },
  "reason": "selected kimi_k2.5 for code_generation exploratory task under budget 2.00 due to lower cost and sufficient quality history; claude_opus set as fallback for escalation",
  "mode": "exploratory"
}
```

### 4.3 Routing Logic (Optimization & Cold‑Start Rules)

**Objective:**

- Minimize `cost_incurred_per_successful_task`.

**Hard Constraints (must never be violated):**

- `quality_score ≥ Q_min`  
- `safety_score ≥ S_min`  
- `latency_ms ≤ L_max`  
- `error_type ∉ { HALLUCINATION.CRITICAL, SAFETY_POLICY_VIOLATION.CRITICAL, POLICY_ERROR.BUDGET_EXCEEDED.CRITICAL }`

**Soft Preferences:**

- Prefer providers whose `known_strengths` match `task_type` / `domain_tags`.  
- Avoid providers whose `known_weaknesses` conflict with the task.  
- For equal quality and safety, prefer lower latency and cost.

**Cold‑Start Provider Rule:**

- When `recent_eval_stats` are missing or insufficient for a provider (e.g., new model), the Router MUST treat it under `routing_mode = exploratory` with:
  - Conservative default constraints on quality/safety (e.g., use high thresholds and rely on Eval Harness to confirm).  
  - A strict **risk budget** (maximum fraction of tasks or budget assigned to this provider).  
- This allows controlled experimentation without over‑committing to untested providers.

### 4.4 Stability & Escalation Policies

Rules:

1. **Default Provider per Context**  
   - Each project/user is assigned a default provider.  
   - Used unless an escalation trigger fires.

2. **Escalation Triggers**  
   - `quality_score < threshold` for N consecutive tasks in a given scenario.  
   - `error_type` in { HALLUCINATION.CRITICAL, SAFETY_POLICY_VIOLATION.CRITICAL }.  
   - `budget_cap` requires cheaper model under exploratory mode.  
   - Explicit user request to switch provider.

3. **Cooldown Period**  
   - After switching providers, the new provider must be used for M tasks before changing again.  
   - Prevents oscillation (thrashing).

4. **Change Logging**  
   - Every provider change logs: `reason`, `severity`, `affected_scenarios`.  
   - Exposed via Observability dashboards.

***

## 5. Observability & Privacy Gate

### 5.1 Responsibility

Observability:

- Ingests traces from Routing, Execution Harness, Vendor Abstraction, Eval Harness.  
- Produces daily, weekly, monthly reports (cost, quality, latency, errors, provider performance).  
- Detects anomalies (quality drops, cost spikes, thrashing, policy violations).

Privacy Gate:

- Scrubs sensitive data from traces before Warm/Cold storage.  
- Enforces training/usage policies per project and provider.  
- Acts as a **passive filter**: transforms traces, does not silently drop them.

### 5.2 Trace Contracts

Three primary trace types:

1. **Routing Trace**

```json
{
  "decision_id": "UUID",
  "timestamp": "...",
  "task_profile": {...},
  "chosen_provider_id": "kimi_k2.5",
  "fallback_provider_id": "claude_opus",
  "mode": "exploratory",
  "reason": "...",
  "severity": "INFO | WARNING | CRITICAL"
}
```

2. **Execution Trace**

```json
{
  "run_id": "RUN_UUID",
  "task_id": "TASK_UUID",
  "provider_id": "kimi_k2.5",
  "request_id": "UUID",
  "latency_ms": 800,
  "tool_calls_count": 7,
  "context_tokens_used": 8500,
  "budget_used": 3.10,
  "budget_cap": 5.00,
  "error": { ... }
}
```

3. **Eval Trace**  
   - As per section 1.5 (Eval Result + context + success_contract).

### 5.3 Privacy Gate Contract

Before any trace is persisted:

- **PII Stripping**  
  - Detect and replace user names, emails, phone numbers, IDs, and sensitive file paths with tokens (e.g., `[USER_NAME]`, `[EMAIL]`, `[FILE_PATH]`).

- **Domain‑Aware Masking**  
  - For `domain_tags` like `medical`, `finance`, apply stronger masking (scrub numeric identifiers, diagnosis codes, account numbers).

- **Training Flag Enforcement**  
  - If `allow_for_training: false`, trace data must not be exported to Cold/Training datasets.

- **Passive Middleware Rule**  
  - The Privacy Gate **MUST NOT** drop or refuse traces except when they violate explicit storage/legal policies.  
  - Any trace rejected must be logged as a `POLICY_ERROR` and surfaced in Observability.  
  - Normally, traces are transformed (scrubbed/masked) and forwarded, not removed.

- **RBAC**  
  - Developers see aggregate metrics and normalized content.  
  - Security/Audit roles may access raw traces in Hot Storage for critical incidents only.

***

## 6. Human‑in‑the‑Loop (HITL) Controls

### 6.1 Responsibility

Humans do not execute tasks; they:

- Adjust Cost Router policies.  
- Calibrate the LLM‑as‑Judge.  
- Evolve Success Contracts.  
- Perform root‑cause analysis on critical failures.

### 6.2 HITL Contracts

1. **Router Policy Override**

```json
{
  "override_id": "UUID",
  "scope": "project_X | global",
  "change": {
    "quality_min": 85,
    "routing_mode": "conservative",
    "default_provider_id": "claude_opus"
  },
  "reason": "project X now handles medical-like tasks; higher quality/safety required",
  "approved_by": "ENGINEER_ID",
  "timestamp": "..."
}
```

2. **Judge Calibration Run**

- Run LLM‑as‑Judge on a fixed Golden Dataset (~100 labeled cases).  
- Compare its scores to ground truth.  
- If deviation > threshold (e.g., 5%), pause judge usage, route evals to humans, and retune prompts/config.

3. **Success Contract Update**

```json
{
  "scenario_id": "SCENARIO_PY_FIX",
  "old_contract": { ... },
  "new_contract": {
    "must_pass_tests": true,
    "must_pass_linter": true,
    "max_tool_calls": 10,
    "allowed_error_types": ["NONE"]
  },
  "reason": "agents were producing unmaintainable code; added linter requirement",
  "approved_by": "ARCHITECT_ID",
  "timestamp": "..."
}
```

4. **Failure Triage Record**

```json
{
  "incident_id": "INCIDENT_UUID",
  "run_id": "RUN_UUID",
  "eval_id": "UUID",
  "classified_root_cause": "TOOL_ERROR.INVALID_TOOL_INPUT",
  "notes": "skill schema incorrect; agent behaved correctly per harness",
  "action_items": ["update tool schema", "add test for missing input field"]
}
```

***

## 7. Layer Dependency Rules

### 7.1 Interaction Diagram (Conceptual)

- **Execution Harness**  
  - Receives: routing decisions from Cost Router, agent/task specs from upper layers.  
  - Sends: unified agent requests to Vendor Abstraction; eval requests to Neutral Eval Harness; execution traces to Observability.

- **Vendor Abstraction Layer**  
  - Receives: agent requests from Execution Harness; Budget Interrupt commands; provider selection hints from Cost Router.  
  - Sends: unified responses to Execution Harness; raw/normalized outputs + errors to Neutral Eval Harness & Observability.

- **Neutral Eval Harness**  
  - Receives: eval inputs from Execution Harness.  
  - Sends: eval results (metrics) to Observability & Cost Router.

- **Cost Routing System**  
  - Receives: provider profiles & aggregated eval stats from Observability.  
  - Sends: routing decisions to Execution Harness & traces to Observability.

- **Observability & Privacy Gate**  
  - Receives: traces from all layers.  
  - Sends: reports/alerts; **no direct execution commands**.

### 7.2 Dependency Constraints

- Layers must not import each other’s code directly; communication is via:
  - JSON message contracts over an internal event bus, or  
  - Narrow REST/GRPC APIs consistent with this spec.

- Components **MUST NOT** rely on blocking synchronous calls for long‑running or multi‑agent tasks; message handlers should be idempotent, with explicit timeouts and retry/backoff strategies.

- Any direct cross‑layer imports/functions should be treated as spec violations and blocked by CI/build rules.

