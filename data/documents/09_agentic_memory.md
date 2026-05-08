# Agentic Memory

## Overview
Agentic memory refers to the memory systems that allow LLM-based agents to maintain, retrieve, and reason over information across tasks, time, and agent interactions. It is more complex than conversational memory because agents must manage state across multiple tool calls, sub-agents, and long-horizon tasks.

## The Four Types of Agentic Memory

### 1. Working Memory (In-Context)
The agent's active scratchpad — the current context window:
- Contains: current task, recent observations, tool outputs, intermediate reasoning
- Bounded by context window size (2K–128K tokens depending on model)
- Fast access, but temporary — lost after the current task
- Managed via: SharedContext objects, LangGraph state, LangChain AgentExecutor state

### 2. Episodic Memory (Short-to-Medium Term)
Records of past agent actions and experiences:
- "I called web_search('flash attention paper') and got result X"
- "Task t2 failed because the database was empty"
- Stored: in a vector database or structured log (PostgreSQL)
- Retrieved: semantic similarity to current task
- Enables: avoid repeating failed approaches, learn from past sessions

### 3. Semantic Memory (Long-term Knowledge)
Factual knowledge the agent has accumulated:
- Domain knowledge (RAG documents, technical facts)
- User preferences and profile information
- Task patterns and templates
- Stored: vector store (pgvector, Pinecone, Qdrant) or knowledge graph
- Retrieved: dense retrieval (embedding similarity) or sparse (BM25)

### 4. Procedural Memory (Skills/Tool Use)
Knowledge about how to accomplish tasks — essentially the agent's tools and prompts:
- Tool specifications and usage patterns
- Prompt templates that work well for specific task types
- Retrieval strategies for different query types
- Stored: as code, prompt templates, or fine-tuned model weights
- Updated: via self-improving prompt loops (like MetaAgent in this system)

## Memory Architecture for Multi-Agent Systems

### Shared vs Private Memory
- **Shared context**: all agents in a pipeline read/write to a common SharedContext object (see this system's implementation)
- **Private memory**: each agent maintains its own episode buffer for self-reflection
- **Global memory**: cross-session persistent store (PostgreSQL + pgvector)

### Memory Handoff Between Agents
In orchestrated multi-agent systems:
1. Orchestrator maintains the master SharedContext
2. Each sub-agent writes its output and citations to SharedContext
3. Downstream agents read from SharedContext (no direct agent-to-agent calls)
4. Memory is accumulated, not replaced

## The MemGPT Architecture (Packer et al., 2023)
MemGPT (Memory-augmented GPT) uses a hierarchical memory system:
- **Main context**: current in-context window (like working memory)
- **External storage**: archival memory (vector store) and recall memory (recent history)
- Agent explicitly manages memory via tools: `memory_insert`, `memory_search`, `memory_append`
- OS-inspired paging: content is swapped in/out of the context window

## Memory Retrieval Strategies

### Dense Retrieval
- Embed memory items and queries using the same embedding model
- Retrieve by cosine similarity (pgvector: `ORDER BY embedding <=> query_embedding LIMIT k`)
- Works well for semantic similarity

### Sparse Retrieval (BM25)
- TF-IDF based keyword matching
- Better for exact phrase recall ("what was the error message?")

### Hybrid Retrieval
- Combine dense + sparse scores (typically 0.7×dense + 0.3×sparse)
- State-of-the-art for most retrieval tasks

## Memory Failure Modes
- **Memory interference**: conflicting memories from different sessions override each other
- **Stale memory**: outdated facts not updated when world changes
- **Retrieval failure**: relevant memory not retrieved due to embedding mismatch
- **Context overflow**: too much retrieved memory crowds out reasoning space
