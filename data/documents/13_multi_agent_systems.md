# Multi-Agent LLM Systems

## What Is a Multi-Agent System?
A collection of LLM instances (agents) with specialised roles that coordinate to complete complex tasks neither could solve alone. Coordination patterns determine how agents communicate, delegate, and share state.

## Coordination Patterns

### Orchestrator-Worker
A central orchestrator decides which workers to invoke and in what order. Workers are specialised (retrieval, critique, code execution). Workers do not call each other directly.
- **Pros**: clear control flow, easy to audit, no circular dependencies
- **Cons**: orchestrator is a bottleneck

### Debate (Society of Mind)
Multiple agents generate independent answers, then critique each other. Final answer is a consensus.
- Used in: Constitutional AI, scalable oversight experiments
- **Pros**: reduces hallucination, catches blind spots
- **Cons**: expensive (multiple full passes)

### Hierarchical Agents
Tree structure: top-level agent delegates to mid-level agents, which delegate to specialists.
- Enables very complex task decomposition
- **Risk**: errors cascade down the tree

### Pipeline (Sequential)
Agent A → Agent B → Agent C. Each agent's output is the next agent's input.
- Simplest to implement
- **Limitation**: no dynamic routing

## Inter-Agent Communication
All inter-agent communication must pass through a well-defined shared state object — agents should not call each other directly. This:
- Makes the system auditable (every state change is logged)
- Prevents circular dependencies
- Allows the orchestrator to intercept and modify messages

## Shared Context Schema
Every agent reads from and writes to a SharedContext:
  - original_query: the user's request
  - sub_tasks: decomposed tasks with dependency graph
  - retrieval_results: chunks retrieved by RAG agent with citations
  - agent_outputs: map of agent_id → output with citations
  - critique_report: per-claim scores and flagged spans
  - final_answer: synthesised response with provenance map
  - tool_call_log: all tool calls with inputs, outputs, latencies
  - policy_violations: budget overflows, policy skips

## Context Window Management
Each agent has a declared token budget. The ContextBudgetManager:
- Tracks consumption per agent
- Triggers CompressionAgent when budget is near
- Logs violations when agents exceed budget

## Failure Modes in Multi-Agent Systems

### Adversarial Cascades
A single agent's hallucinated output propagates to downstream agents who treat it as ground truth. Mitigation: CritiqueAgent reviews all outputs; SynthesisAgent resolves flagged claims.

### Tool Call Amplification
One agent triggers a tool that triggers another tool in a loop. Mitigation: max_tool_retries=2, explicit fallback logic in code.

### Context Poisoning
Malicious content in a retrieved document overwrites agent instructions. Mitigation: instruction hierarchy, output schema validation, separate channels for user data and system instructions.

### Deadlock
Two agents waiting for each other's output. Mitigation: dependency graph enforced by orchestrator; agents declare dependencies explicitly.

## Production Considerations
- **Observability**: structured logging with job_id, agent_id, input/output hashes, latency
- **Reproducibility**: store exact prompts + tool inputs to replay any run
- **Rate limiting**: Groq free tier: ~30 RPM; use exponential backoff
- **Cost management**: count tokens per agent per run; alert on budget violations
