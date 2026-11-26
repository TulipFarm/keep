-- Chunk vectors kept with the ledger, so re-sending unchanged text (a
-- retried batch, a replaced source whose episodes mostly stayed the same)
-- reuses its vector instead of embedding it again. `digest` covers the
-- embedding model and the exact text. A revision that removes a chunk sets
-- `removed_at`; the row is swept after a grace period. Forgetting a source
-- or corpus deletes its rows at once.
CREATE TABLE knowledge_vectors (
    corpus text NOT NULL,
    point_id text NOT NULL,
    source_id text NOT NULL,
    digest text NOT NULL,
    vector real[] NOT NULL,
    stored_at timestamptz NOT NULL,
    removed_at timestamptz,
    PRIMARY KEY (corpus, point_id)
);
CREATE INDEX knowledge_vectors_source ON knowledge_vectors (corpus, source_id);
CREATE INDEX knowledge_vectors_removed ON knowledge_vectors (removed_at)
    WHERE removed_at IS NOT NULL;

-- Claiming one source's jobs and dropping a purged episode's jobs look jobs
-- up by target; only unfinished ones matter.
CREATE INDEX IF NOT EXISTS knowledge_jobs_open_target ON knowledge_jobs (corpus, target)
    WHERE state NOT IN ('succeeded', 'failed');
