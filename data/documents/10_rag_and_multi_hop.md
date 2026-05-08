# Retrieval-Augmented Generation (RAG) and Multi-Hop Reasoning

## What Is RAG?
RAG (Lewis et al., 2020) combines parametric memory (LLM weights) with non-parametric memory (retrieved documents) to produce grounded, factual responses.

Pipeline:
  Query → Encoder → Dense Retrieval → Top-k Chunks → LLM Generation with Context

## Why RAG?
- LLMs hallucinate when asked about facts not in training data
- LLMs cannot access events after their training cutoff
- RAG grounds generation in retrieved evidence — answers are verifiable and citable
- Cheaper to update knowledge base than to retrain the LLM

## RAG Components

### 1. Document Indexing
- Split documents into overlapping chunks (e.g., 512 tokens, 64 token overlap)
- Embed each chunk with an embedding model (OpenAI text-embedding-ada-002, BGE, E5)
- Store vectors in a vector database (pgvector, Pinecone, Qdrant, FAISS)

### 2. Query Encoding
- Embed the user query using the same embedding model
- Optionally: rephrase query via HyDE (hypothetical document embeddings) or query expansion

### 3. Retrieval
- k-nearest neighbours in embedding space (cosine similarity)
- pgvector: `SELECT * FROM chunks ORDER BY embedding <=> $1 LIMIT 5`

### 4. Generation
- Concatenate query + retrieved chunks as context
- LLM generates answer conditioned on both

## Standard RAG Limitations
- Single-hop: one retrieval step then generate
- Fails on complex questions requiring multi-step reasoning
- E.g.: "Who is the CEO of the company that made the model used in this system?" requires: (1) find the model, (2) find the company, (3) find the CEO

## Multi-Hop RAG

### What It Is
Multi-hop RAG performs retrieval iteratively:
1. **Hop 1**: Retrieve documents relevant to the original query
2. **Intermediate reasoning**: form an intermediate conclusion from hop 1 results
3. **Hop 2**: Use the intermediate conclusion to formulate a new query and retrieve more documents
4. **Final generation**: combine all retrieved evidence to produce the final answer

### Why It Matters
For questions requiring chaining of facts, single-hop retrieval misses the second-step evidence. Multi-hop retrieval ensures all evidence is gathered before answering.

## Advanced RAG Techniques

### HyDE (Hypothetical Document Embeddings)
1. LLM generates a hypothetical answer to the query
2. Embed the hypothetical answer (instead of the query)
3. Retrieve documents similar to the hypothetical answer
- Improves retrieval for queries that are semantically far from the answer style

### RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval)
- Build a tree of summaries: leaf nodes = raw chunks, internal nodes = summaries of clusters
- Query at multiple tree levels for comprehensive coverage

### Self-RAG
- LLM decides whether retrieval is needed for each segment
- Generates reflection tokens: [Retrieve], [ISREL], [ISSUP], [ISUSE]
- Only retrieves when necessary, then critiques retrieved content quality

## Citation Requirements in This System
The RAGAgent in this system must:
1. Retrieve at least 2 chunks (multi-hop: hop 1 then hop 2 with updated query)
2. Tag each sentence with [chunk_id] citations
3. Include at least 2 distinct chunk_ids in citations
4. Single-hop retrieval results fail schema validation
