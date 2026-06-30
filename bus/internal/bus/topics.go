// SAHA – Event Bus Topic Definitions (Go)
// These constants MUST mirror saha/event_bus/topics.py exactly.
// Switching backends (Redis → Kafka) means changing only the adapter,
// never these topic strings.
package bus

const (
	TopicAgentRequests    = "SAHA/agent_requests"
	TopicProviderResponses = "SAHA/provider_responses"
	TopicEvalInputs       = "SAHA/eval_inputs"
	TopicEvalResults      = "SAHA/eval_results"
	TopicBudgetInterrupts = "SAHA/budget_interrupts"
)

// AllTopics enumerates every known SAHA topic.
var AllTopics = []string{
	TopicAgentRequests,
	TopicProviderResponses,
	TopicEvalInputs,
	TopicEvalResults,
	TopicBudgetInterrupts,
}
