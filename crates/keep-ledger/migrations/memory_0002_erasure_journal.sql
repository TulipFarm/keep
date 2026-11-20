-- Content-free record of each forget decision. It may live in a different
-- database from the ledger so that it survives a ledger restore.

CREATE TABLE memory_erasure_journal (
    seq         bigserial PRIMARY KEY,
    tree_id     text NOT NULL,
    record      jsonb NOT NULL
);
CREATE INDEX memory_erasure_journal_tree ON memory_erasure_journal (tree_id, seq);
