-- Small host-defined string map per source, such as a document URL or id.
ALTER TABLE knowledge_sources
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
