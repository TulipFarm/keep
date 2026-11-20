-- Knowledge ledger: sources, episodes, chunks, entities, bi-temporal facts,
-- evidence, jobs, index cache and the erasure journal. Every row carries the
-- corpus and the partition (visibility label) inherited from its source.

CREATE TABLE knowledge_sources (
    corpus text NOT NULL,
    id text NOT NULL,
    partition_label text NOT NULL,
    collection text NOT NULL,
    title text NOT NULL,
    kind text NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (corpus, id)
);

CREATE TABLE knowledge_episodes (
    corpus text NOT NULL,
    id text NOT NULL,
    source_id text NOT NULL,
    partition_label text NOT NULL,
    ordinal integer NOT NULL,
    speaker text NOT NULL,
    occurred_at timestamptz,
    recorded_at timestamptz NOT NULL,
    text text NOT NULL,
    content_hash text NOT NULL,
    PRIMARY KEY (corpus, id),
    FOREIGN KEY (corpus, source_id) REFERENCES knowledge_sources (corpus, id) ON DELETE CASCADE
);
CREATE INDEX knowledge_episodes_source ON knowledge_episodes (corpus, source_id);

CREATE TABLE knowledge_chunks (
    corpus text NOT NULL,
    id text NOT NULL,
    episode_id text NOT NULL,
    source_id text NOT NULL,
    partition_label text NOT NULL,
    ordinal integer NOT NULL,
    char_start integer NOT NULL,
    char_end integer NOT NULL,
    speaker text NOT NULL,
    text text NOT NULL,
    occurred_at timestamptz,
    recorded_at timestamptz NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (corpus, id),
    FOREIGN KEY (corpus, episode_id) REFERENCES knowledge_episodes (corpus, id) ON DELETE CASCADE
);
CREATE INDEX knowledge_chunks_tsv ON knowledge_chunks USING gin (tsv);
CREATE INDEX knowledge_chunks_partition ON knowledge_chunks (corpus, partition_label);
CREATE INDEX knowledge_chunks_source ON knowledge_chunks (corpus, source_id);

CREATE TABLE knowledge_entities (
    corpus text NOT NULL,
    id text NOT NULL,
    partition_label text NOT NULL,
    name text NOT NULL,
    norm_name text NOT NULL,
    kind text NOT NULL,
    PRIMARY KEY (corpus, id)
);
CREATE INDEX knowledge_entities_name ON knowledge_entities (corpus, partition_label, norm_name);

CREATE TABLE knowledge_entity_aliases (
    corpus text NOT NULL,
    entity_id text NOT NULL,
    alias text NOT NULL,
    norm_alias text NOT NULL,
    source_id text NOT NULL,
    PRIMARY KEY (corpus, entity_id, alias, source_id),
    FOREIGN KEY (corpus, entity_id) REFERENCES knowledge_entities (corpus, id) ON DELETE CASCADE
);
CREATE INDEX knowledge_aliases_norm ON knowledge_entity_aliases (corpus, norm_alias);
CREATE INDEX knowledge_aliases_source ON knowledge_entity_aliases (corpus, source_id);

CREATE TABLE knowledge_facts (
    corpus text NOT NULL,
    id text NOT NULL,
    partition_label text NOT NULL,
    subject text NOT NULL,
    predicate text NOT NULL,
    object text NOT NULL,
    object_entity text,
    text text NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (corpus, id)
);
CREATE INDEX knowledge_facts_tsv ON knowledge_facts USING gin (tsv);
CREATE INDEX knowledge_facts_subject ON knowledge_facts (corpus, partition_label, subject, predicate);
CREATE INDEX knowledge_facts_object ON knowledge_facts (corpus, object_entity);

-- Immutable versions. A newer version sets invalidated_at on the previous
-- one (system time); valid_from/valid_to describe world time.
CREATE TABLE knowledge_fact_versions (
    corpus text NOT NULL,
    fact_id text NOT NULL,
    version integer NOT NULL,
    valid_from timestamptz,
    valid_to timestamptz,
    valid_from_basis text NOT NULL,
    status text NOT NULL,
    recorded_at timestamptz NOT NULL,
    invalidated_at timestamptz,
    cause text,
    PRIMARY KEY (corpus, fact_id, version),
    FOREIGN KEY (corpus, fact_id) REFERENCES knowledge_facts (corpus, id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX knowledge_fact_versions_current
    ON knowledge_fact_versions (corpus, fact_id) WHERE invalidated_at IS NULL;
CREATE INDEX knowledge_fact_versions_cause ON knowledge_fact_versions (corpus, cause);

CREATE TABLE knowledge_evidence (
    corpus text NOT NULL,
    id text NOT NULL,
    fact_id text NOT NULL,
    episode_id text NOT NULL,
    source_id text NOT NULL,
    partition_label text NOT NULL,
    char_start integer NOT NULL,
    char_end integer NOT NULL,
    quote text NOT NULL,
    extractor text NOT NULL,
    recorded_at timestamptz NOT NULL,
    PRIMARY KEY (corpus, id),
    FOREIGN KEY (corpus, fact_id) REFERENCES knowledge_facts (corpus, id) ON DELETE CASCADE,
    FOREIGN KEY (corpus, episode_id) REFERENCES knowledge_episodes (corpus, id) ON DELETE CASCADE
);
CREATE INDEX knowledge_evidence_fact ON knowledge_evidence (corpus, fact_id);
CREATE INDEX knowledge_evidence_source ON knowledge_evidence (corpus, source_id);

CREATE TABLE knowledge_jobs (
    id bigserial PRIMARY KEY,
    corpus text NOT NULL,
    kind text NOT NULL,
    target text NOT NULL,
    state text NOT NULL,
    attempts integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL,
    next_attempt_at timestamptz NOT NULL,
    lease bigint,
    last_error text
);
CREATE INDEX knowledge_jobs_due ON knowledge_jobs (corpus, state, next_attempt_at);
CREATE SEQUENCE knowledge_job_lease_seq;

CREATE TABLE knowledge_partitions (
    corpus text NOT NULL,
    partition_label text NOT NULL,
    generation bigint NOT NULL,
    PRIMARY KEY (corpus, partition_label)
);

CREATE TABLE knowledge_index_cache (
    corpus text NOT NULL,
    partition_label text NOT NULL,
    generation bigint NOT NULL,
    body text NOT NULL,
    PRIMARY KEY (corpus, partition_label)
);

-- Content-free: digests and counts only.
CREATE TABLE knowledge_erasure_journal (
    seq bigserial PRIMARY KEY,
    corpus_digest text NOT NULL,
    target_kind text NOT NULL,
    target_digest text NOT NULL,
    erased_at timestamptz NOT NULL,
    episodes bigint NOT NULL,
    chunks bigint NOT NULL,
    facts bigint NOT NULL
);
