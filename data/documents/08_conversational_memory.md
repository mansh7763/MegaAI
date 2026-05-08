# Conversational Memory in LLMs

## The Problem
LLMs have no inherent persistent memory — each API call is stateless. Conversational memory is the set of techniques used to give LLMs the appearance of memory across a conversation or across sessions.

## Types of Conversational Memory

### 1. Buffer Memory (Full Context Window)
Store the complete conversation history as a list of messages:
  messages = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

**Pros**: Perfect recall, simple
**Cons**: Grows without bound; eventually exceeds context window; expensive at long lengths

### 2. Sliding Window / Windowed Buffer
Keep only the last k messages:
  context = messages[-k:]

**Pros**: Bounded cost
**Cons**: Loses early context (e.g., user's name, initial goals)

### 3. Summarisation Memory
Periodically summarise older conversation turns into a compressed summary:
  if len(messages) > threshold:
      summary = llm.summarise(messages[:-recent_k])
      context = [summary_message] + messages[-recent_k:]

**Pros**: Retains key information in compressed form
**Cons**: Lossy; summariser may miss important details

### 4. Entity Memory
Extract and maintain a structured "entity store" — a key-value map of entities mentioned:
  entities = {"user_name": "Alice", "project": "MegaAI", "preference": "Python"}

**Pros**: Precise recall of specific facts
**Cons**: Requires extraction step; misses relational/narrative context

### 5. Vector Store Memory (Long-term Semantic Memory)
Embed all conversation turns and store in a vector database:
- At each turn, retrieve top-k semantically similar past turns
- Inject retrieved turns into context

**Pros**: Can recall relevant content from very long histories
**Cons**: Requires embedding + retrieval infrastructure; may miss exact wording

## Memory in Production Systems

### LangChain Memory Abstractions
- `ConversationBufferMemory`: full history
- `ConversationSummaryMemory`: summarises old turns
- `ConversationBufferWindowMemory`: sliding window
- `VectorStoreRetrieverMemory`: semantic retrieval
- `ConversationKGMemory`: knowledge graph of entities

### Context Window Management Strategy
Modern long-context models (Gemini 1.5 Pro: 1M tokens, Claude 3.5: 200K tokens) reduce the need for complex memory strategies for single sessions, but cross-session memory still requires persistence mechanisms.

## Episodic vs Semantic Memory
- **Episodic**: specific events/conversations with temporal context ("User asked about RAG on May 5")
- **Semantic**: general facts without temporal context ("User is a Python developer")

Both types benefit from hybrid storage: structured (SQL/Redis) for semantic facts, vector store for episodic retrieval.

## Memory Injection Patterns
1. **Prefix injection**: prepend memory to system prompt
2. **Inline injection**: insert relevant memory as a "memory bank" message
3. **RAG injection**: retrieve and format as retrieved context blocks
4. **Tool-based**: give the LLM a `recall_memory(query)` tool to query its own memory store
