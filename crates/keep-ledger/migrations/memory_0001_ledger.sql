-- Memory ledger: trees, files, immutable revisions with their entries,
-- and idempotency receipts. Trash is a file with deleted_at set.

CREATE TABLE memory_trees (
    tree_id     text PRIMARY KEY,
    record      jsonb NOT NULL
);

CREATE TABLE memory_files (
    memory_id   text PRIMARY KEY,
    tree_id     text NOT NULL REFERENCES memory_trees (tree_id),
    path        text NOT NULL,
    deleted_at  timestamptz,
    record      jsonb NOT NULL
);
CREATE INDEX memory_files_tree_path ON memory_files (tree_id, path COLLATE "C", memory_id COLLATE "C");

CREATE VIEW memory_trash AS
    SELECT memory_id, tree_id, path, deleted_at FROM memory_files WHERE deleted_at IS NOT NULL;

CREATE TABLE memory_revisions (
    revision_id text PRIMARY KEY,
    memory_id   text NOT NULL,
    tree_id     text NOT NULL,
    seq         bigint NOT NULL,
    parents     jsonb NOT NULL,
    change      text NOT NULL,
    actor       text NOT NULL,
    harness     text,
    created_at  timestamptz NOT NULL,
    version     text NOT NULL,
    meta        jsonb NOT NULL,
    UNIQUE (memory_id, seq)
);

-- Canonical entries of each revision, in document order.
CREATE TABLE memory_entries (
    revision_id text NOT NULL REFERENCES memory_revisions (revision_id) ON DELETE CASCADE,
    ord         integer NOT NULL,
    entry_id    text NOT NULL,
    kind        text NOT NULL,
    body        text NOT NULL,
    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,
    lineage     jsonb NOT NULL,
    annotation  jsonb NOT NULL,
    PRIMARY KEY (revision_id, ord)
);
CREATE INDEX memory_entries_entry ON memory_entries (entry_id);

CREATE TABLE memory_receipts (
    tree_id     text NOT NULL,
    principal   text NOT NULL,
    key         text NOT NULL,
    memory_id   text,
    record      jsonb NOT NULL,
    PRIMARY KEY (tree_id, principal, key)
);
CREATE INDEX memory_receipts_memory ON memory_receipts (memory_id);
