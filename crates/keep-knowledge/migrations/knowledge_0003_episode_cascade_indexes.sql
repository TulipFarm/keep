-- Deleting an episode cascades to its chunks and evidence by
-- (corpus, episode_id). Without these indexes each cascaded episode scans
-- the corpus's chunks and evidence, so forgetting a source with many
-- episodes took minutes and held the ledger connection throughout.
CREATE INDEX IF NOT EXISTS knowledge_chunks_episode ON knowledge_chunks (corpus, episode_id);
CREATE INDEX IF NOT EXISTS knowledge_evidence_episode ON knowledge_evidence (corpus, episode_id);
