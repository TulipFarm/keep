-- Finding due corpora and claiming their jobs read only unfinished jobs.
-- `knowledge_jobs_due` leads with corpus and holds every finished job too,
-- so the worker's due-corpora scan, run every few seconds, read the whole
-- index. This one holds unfinished jobs only.
--
-- Migrations run in one transaction, so these builds block writes to
-- knowledge_jobs while they run. On a large table, create them first with
-- CREATE INDEX CONCURRENTLY and the same definitions; IF NOT EXISTS then
-- skips them here.
CREATE INDEX IF NOT EXISTS knowledge_jobs_open_due ON knowledge_jobs (corpus, next_attempt_at)
    WHERE state IN ('pending', 'retry_scheduled', 'running');
-- Pruning walks finished jobs oldest first.
CREATE INDEX IF NOT EXISTS knowledge_jobs_finished ON knowledge_jobs (state, next_attempt_at)
    WHERE state IN ('succeeded', 'failed');
