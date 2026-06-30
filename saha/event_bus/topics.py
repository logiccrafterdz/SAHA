"""
SAHA – Event Bus topic definitions.
Topics are stable SAHA contracts; switching from Redis → Kafka
requires only changing the adapter, never the topic names.
"""

# Publisher: Execution Harness  → Subscriber: Vendor Abstraction
AGENT_REQUESTS = "SAHA/agent_requests"

# Publisher: Vendor Abstraction → Subscriber: Execution Harness
PROVIDER_RESPONSES = "SAHA/provider_responses"

# Publisher: Execution Harness  → Subscriber: Eval Harness
EVAL_INPUTS = "SAHA/eval_inputs"

# Publisher: Eval Harness       → Subscriber: Observability / Cost Router
EVAL_RESULTS = "SAHA/eval_results"

# Publisher: Execution Harness  → Subscriber: Vendor Abstraction
BUDGET_INTERRUPTS = "SAHA/budget_interrupts"

# Publisher: Observability (AnomalyDetector) → Subscriber: Routing, HITL, dashboards
ANOMALY_ALERTS = "SAHA/anomaly_alerts"

# All known topics
ALL_TOPICS = [
    AGENT_REQUESTS,
    PROVIDER_RESPONSES,
    EVAL_INPUTS,
    EVAL_RESULTS,
    BUDGET_INTERRUPTS,
    ANOMALY_ALERTS,
]
