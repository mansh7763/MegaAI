CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    final_answer TEXT,
    context_snapshot JSONB,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    latency_ms INTEGER,
    token_count INTEGER,
    policy_violations JSONB DEFAULT '[]',
    payload JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tool_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    tool_name VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    latency_ms INTEGER,
    accepted BOOLEAN,
    retry_count INTEGER DEFAULT 0,
    error_code VARCHAR(50),
    input_payload JSONB,
    output_payload JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_cases INTEGER,
    summary JSONB,
    triggered_by VARCHAR(50) DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES eval_runs(id) ON DELETE CASCADE,
    category VARCHAR(20) NOT NULL,
    query TEXT NOT NULL,
    ground_truth TEXT,
    agent_prompts JSONB,
    tool_calls JSONB,
    agent_outputs JSONB,
    final_answer TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES eval_cases(id) ON DELETE CASCADE,
    dimension VARCHAR(50) NOT NULL,
    score FLOAT NOT NULL,
    justification TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_rewrites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id VARCHAR(100) NOT NULL,
    old_text TEXT NOT NULL,
    new_text TEXT NOT NULL,
    unified_diff TEXT NOT NULL,
    justification TEXT NOT NULL,
    proposed_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending',
    decided_at TIMESTAMPTZ,
    decision VARCHAR(10),
    delta_scores JSONB,
    run_id UUID REFERENCES eval_runs(id)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    source_file VARCHAR(200),
    chunk_index INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_agent_logs_job_id ON agent_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_tool_logs_job_id ON tool_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_eval_cases_run_id ON eval_cases(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_case_id ON eval_scores(case_id);
CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_status ON prompt_rewrites(status);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
