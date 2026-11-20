//! Postgres storage for Keep's MEMORY primitive.
//!
//! [`PgMemoryLedger`] implements [`MemoryLedger`] on Postgres. Each
//! transaction is a real database transaction: the tree row is locked with
//! `SELECT … FOR UPDATE`, so concurrent writers to one tree queue behind each
//! other and compare-and-swap checks see committed state.
//!
//! Revisions are stored in `memory_revisions` with their canonical entries
//! in `memory_entries`; deleting a revision cascades to its entries. Trashed
//! files keep `deleted_at` and are exposed by the `memory_trash` view.
//!
//! [`PgErasureJournal`] keeps content-free forget records. Point it at a
//! different database (or at least a different backup schedule) from the
//! ledger if it must survive a ledger restore, and say so with
//! `independent = true`.
//!
//! Migrations are embedded and applied on connect. Every table name starts
//! with `memory_`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use keep_memory::model::{
    Annotation, ChangeKind, Entry, ErasureRecord, FileMeta, FileRecord, Lineage, Receipt, Revision,
    TreeRecord,
};
use keep_memory::{EntryKind, ErasureJournal, LedgerTx, MemoryError, MemoryLedger, Result};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_postgres::types::{Json, ToSql};
use tokio_postgres::{Client, Row, Statement};

const LEDGER_MIGRATIONS: &[(&str, &str)] = &[(
    "memory_0001_ledger",
    include_str!("../migrations/memory_0001_ledger.sql"),
)];
const JOURNAL_MIGRATIONS: &[(&str, &str)] = &[(
    "memory_0002_erasure_journal",
    include_str!("../migrations/memory_0002_erasure_journal.sql"),
)];

fn db(e: tokio_postgres::Error) -> MemoryError {
    MemoryError::Storage(format!("postgres: {e}"))
}

fn decode(e: serde_json::Error) -> MemoryError {
    MemoryError::Storage(format!("postgres: bad stored value: {e}"))
}

struct Conn {
    client: Client,
    /// A transaction was left open by a dropped handle and must be rolled
    /// back before the next one starts.
    open: bool,
    statements: HashMap<String, Statement>,
}

type PgResult<T> = std::result::Result<T, tokio_postgres::Error>;

/// Open a connection, optionally pinned to `schema` (created if missing),
/// and apply `migrations`.
async fn connect(url: &str, schema: Option<&str>, migrations: &[(&str, &str)]) -> Result<Client> {
    let client = keep_postgres::connect(url)
        .await
        .map_err(|e| MemoryError::Storage(format!("postgres: {e}")))?;
    if let Some(schema) = schema {
        if schema.is_empty()
            || !schema
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            return Err(MemoryError::InvalidRequest(format!(
                "schema name {schema:?} must be lowercase letters, digits and underscores"
            )));
        }
        client
            .batch_execute(&format!(
                "CREATE SCHEMA IF NOT EXISTS {schema}; SET search_path TO {schema}"
            ))
            .await
            .map_err(db)?;
    }
    migrate(&client, migrations).await?;
    Ok(client)
}

async fn migrate(client: &Client, migrations: &[(&str, &str)]) -> Result<()> {
    client
        .batch_execute(
            "CREATE TABLE IF NOT EXISTS memory_schema_migrations (
                version text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )",
        )
        .await
        .map_err(db)?;
    for (version, sql) in migrations {
        client.batch_execute("BEGIN").await.map_err(db)?;
        let result = async {
            // Serialise concurrent migrators of the same schema.
            client
                .execute(
                    "SELECT pg_advisory_xact_lock(hashtext(current_schema() || '/keep-memory'))",
                    &[],
                )
                .await?;
            let done = client
                .query_opt(
                    "SELECT 1 FROM memory_schema_migrations WHERE version = $1",
                    &[version],
                )
                .await?
                .is_some();
            if !done {
                client.batch_execute(sql).await?;
                client
                    .execute(
                        "INSERT INTO memory_schema_migrations (version) VALUES ($1)",
                        &[version],
                    )
                    .await?;
            }
            client.batch_execute("COMMIT").await
        }
        .await;
        if let Err(e) = result {
            let _ = client.batch_execute("ROLLBACK").await;
            return Err(db(e));
        }
    }
    Ok(())
}

/// The Postgres ledger. One connection; transactions are serialised
/// through it.
#[derive(Clone)]
pub struct PgMemoryLedger {
    conn: Arc<Mutex<Conn>>,
    url: Arc<str>,
    schema: Option<Arc<str>>,
}

impl PgMemoryLedger {
    /// Connect and apply migrations in the connection's default schema.
    pub async fn connect(url: &str) -> Result<Self> {
        Self::connect_in_schema(url, None).await
    }

    /// Connect using `schema` for every table, creating it if needed.
    pub async fn connect_in_schema(url: &str, schema: Option<&str>) -> Result<Self> {
        let client = connect(url, schema, LEDGER_MIGRATIONS).await?;
        Ok(Self {
            conn: Arc::new(Mutex::new(Conn {
                client,
                open: false,
                statements: HashMap::new(),
            })),
            url: url.into(),
            schema: schema.map(Into::into),
        })
    }
}

#[async_trait]
impl MemoryLedger for PgMemoryLedger {
    async fn begin(&self) -> Result<Box<dyn LedgerTx + '_>> {
        let mut conn = self.conn.clone().lock_owned().await;
        // Reopen a connection the server or network closed, so the ledger
        // recovers after a database outage. Prepared statements and any
        // open transaction died with the old connection.
        if conn.client.is_closed() {
            conn.client = connect(&self.url, self.schema.as_deref(), LEDGER_MIGRATIONS).await?;
            conn.statements.clear();
            conn.open = false;
        }
        if conn.open {
            conn.client.batch_execute("ROLLBACK").await.map_err(db)?;
            conn.open = false;
        }
        conn.client.batch_execute("BEGIN").await.map_err(db)?;
        conn.open = true;
        Ok(Box::new(PgTx { conn: Some(conn) }))
    }
}

struct PgTx {
    conn: Option<OwnedMutexGuard<Conn>>,
}

impl Drop for PgTx {
    fn drop(&mut self) {
        // Roll back an uncommitted transaction now if a runtime is available;
        // otherwise `begin` does it before the next transaction.
        if let Some(mut conn) = self.conn.take()
            && conn.open
            && let Ok(handle) = tokio::runtime::Handle::try_current()
        {
            handle.spawn(async move {
                if conn.client.batch_execute("ROLLBACK").await.is_ok() {
                    conn.open = false;
                }
            });
        }
    }
}

impl PgTx {
    /// Prepare `sql` once per connection and reuse it: every prepare is a
    /// round trip.
    async fn stmt(&mut self, sql: &str) -> std::result::Result<Statement, tokio_postgres::Error> {
        let conn = self.conn.as_mut().expect("transaction is live");
        if let Some(s) = conn.statements.get(sql) {
            return Ok(s.clone());
        }
        let s = conn.client.prepare(sql).await?;
        conn.statements.insert(sql.to_owned(), s.clone());
        Ok(s)
    }

    fn client(&self) -> &Client {
        &self.conn.as_ref().expect("transaction is live").client
    }

    async fn q(&mut self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> PgResult<Vec<Row>> {
        let s = self.stmt(sql).await?;
        self.client().query(&s, params).await
    }

    async fn q_opt(&mut self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> PgResult<Option<Row>> {
        let s = self.stmt(sql).await?;
        self.client().query_opt(&s, params).await
    }

    async fn exec(&mut self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> PgResult<u64> {
        let s = self.stmt(sql).await?;
        self.client().execute(&s, params).await
    }

    async fn load_entries(&mut self, revision_ids: &[String]) -> Result<Vec<(String, Entry)>> {
        let rows = self.q(
                "SELECT revision_id, entry_id, kind, body, created_at, updated_at, lineage, annotation
                 FROM memory_entries WHERE revision_id = ANY($1) ORDER BY revision_id, ord",
                &[&revision_ids],
            )
            .await
            .map_err(db)?;
        rows.iter()
            .map(|r| {
                let kind: String = r.get(2);
                let Json(lineage): Json<Lineage> = r.get(6);
                let Json(annotation): Json<Annotation> = r.get(7);
                Ok((
                    r.get(0),
                    Entry {
                        entry_id: r.get(1),
                        kind: serde_json::from_value::<EntryKind>(kind.into()).map_err(decode)?,
                        text: r.get(3),
                        created_at: r.get(4),
                        updated_at: r.get(5),
                        lineage,
                        annotation,
                    },
                ))
            })
            .collect()
    }

    async fn load_revisions(&mut self, rows: Vec<Row>) -> Result<Vec<Revision>> {
        let mut out: Vec<Revision> = rows
            .iter()
            .map(|r| {
                let change: String = r.get(5);
                let Json(parents): Json<Vec<String>> = r.get(4);
                let Json(meta): Json<FileMeta> = r.get(10);
                let seq: i64 = r.get(3);
                Ok(Revision {
                    revision_id: r.get(0),
                    memory_id: r.get(1),
                    tree_id: r.get(2),
                    seq: seq as u64,
                    parents,
                    change: ChangeKind::parse(&change).ok_or_else(|| {
                        MemoryError::Storage(format!("postgres: unknown change {change:?}"))
                    })?,
                    actor: r.get(6),
                    harness: r.get(7),
                    created_at: r.get(8),
                    version: r.get(9),
                    meta,
                    entries: Vec::new(),
                })
            })
            .collect::<Result<_>>()?;
        let ids: Vec<String> = out.iter().map(|r| r.revision_id.clone()).collect();
        let mut entries = self.load_entries(&ids).await?.into_iter().peekable();
        // Both lists are needed in revision order; entries come sorted by
        // revision id, so index revisions by id.
        let mut index: Vec<usize> = (0..out.len()).collect();
        index.sort_by(|a, b| out[*a].revision_id.cmp(&out[*b].revision_id));
        for i in index {
            while let Some((rid, _)) = entries.peek() {
                if *rid != out[i].revision_id {
                    break;
                }
                let (_, e) = entries.next().expect("peeked");
                out[i].entries.push(e);
            }
        }
        Ok(out)
    }
}

const REVISION_COLUMNS: &str = "revision_id, memory_id, tree_id, seq, parents, change, actor, harness, created_at, version, meta";

#[async_trait]
impl LedgerTx for PgTx {
    async fn tree(&mut self, tree_id: &str) -> Result<Option<TreeRecord>> {
        let row = self
            .q_opt(
                "SELECT record FROM memory_trees WHERE tree_id = $1 FOR UPDATE",
                &[&tree_id],
            )
            .await
            .map_err(db)?;
        Ok(row.map(|r| r.get::<_, Json<TreeRecord>>(0).0))
    }

    async fn put_tree(&mut self, tree: &TreeRecord) -> Result<()> {
        self.exec(
            "INSERT INTO memory_trees (tree_id, record) VALUES ($1, $2)
                 ON CONFLICT (tree_id) DO UPDATE SET record = EXCLUDED.record",
            &[&tree.tree_id, &Json(tree)],
        )
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn files(&mut self, tree_id: &str, include_trashed: bool) -> Result<Vec<FileRecord>> {
        let rows = self
            .q(
                "SELECT record FROM memory_files
                 WHERE tree_id = $1 AND ($2 OR deleted_at IS NULL)
                 ORDER BY path COLLATE \"C\", memory_id COLLATE \"C\"",
                &[&tree_id, &include_trashed],
            )
            .await
            .map_err(db)?;
        Ok(rows
            .iter()
            .map(|r| r.get::<_, Json<FileRecord>>(0).0)
            .collect())
    }

    async fn file(&mut self, memory_id: &str) -> Result<Option<FileRecord>> {
        let row = self
            .q_opt(
                "SELECT record FROM memory_files WHERE memory_id = $1",
                &[&memory_id],
            )
            .await
            .map_err(db)?;
        Ok(row.map(|r| r.get::<_, Json<FileRecord>>(0).0))
    }

    async fn put_file(&mut self, file: &FileRecord) -> Result<()> {
        self.exec(
            "INSERT INTO memory_files (memory_id, tree_id, path, deleted_at, record)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (memory_id) DO UPDATE SET
                    tree_id = EXCLUDED.tree_id, path = EXCLUDED.path,
                    deleted_at = EXCLUDED.deleted_at, record = EXCLUDED.record",
            &[
                &file.memory_id,
                &file.tree_id,
                &file.meta.path,
                &file.deleted_at,
                &Json(file),
            ],
        )
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn purge_file(&mut self, memory_id: &str) -> Result<()> {
        self.exec(
            "DELETE FROM memory_revisions WHERE memory_id = $1",
            &[&memory_id],
        )
        .await
        .map_err(db)?;
        self.exec(
            "DELETE FROM memory_files WHERE memory_id = $1",
            &[&memory_id],
        )
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn put_revision(&mut self, rev: &Revision) -> Result<()> {
        let seq = i64::try_from(rev.seq)
            .map_err(|_| MemoryError::Storage("revision sequence overflow".into()))?;
        self.exec(
            &format!(
                "INSERT INTO memory_revisions ({REVISION_COLUMNS})
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                 ON CONFLICT (revision_id) DO UPDATE SET
                    memory_id = EXCLUDED.memory_id, tree_id = EXCLUDED.tree_id,
                    seq = EXCLUDED.seq, parents = EXCLUDED.parents, change = EXCLUDED.change,
                    actor = EXCLUDED.actor, harness = EXCLUDED.harness,
                    created_at = EXCLUDED.created_at, version = EXCLUDED.version,
                    meta = EXCLUDED.meta"
            ),
            &[
                &rev.revision_id,
                &rev.memory_id,
                &rev.tree_id,
                &seq,
                &Json(&rev.parents),
                &rev.change.as_str(),
                &rev.actor,
                &rev.harness,
                &rev.created_at,
                &rev.version,
                &Json(&rev.meta),
            ],
        )
        .await
        .map_err(db)?;
        self.exec(
            "DELETE FROM memory_entries WHERE revision_id = $1",
            &[&rev.revision_id],
        )
        .await
        .map_err(db)?;
        if rev.entries.is_empty() {
            return Ok(());
        }
        let mut ords = Vec::new();
        let mut ids = Vec::new();
        let mut kinds = Vec::new();
        let mut bodies = Vec::new();
        let mut created: Vec<DateTime<Utc>> = Vec::new();
        let mut updated: Vec<DateTime<Utc>> = Vec::new();
        let mut lineages = Vec::new();
        let mut annotations = Vec::new();
        for (i, e) in rev.entries.iter().enumerate() {
            ords.push(i32::try_from(i).unwrap_or(i32::MAX));
            ids.push(e.entry_id.as_str());
            let kind = serde_json::to_value(e.kind).map_err(decode)?;
            kinds.push(kind.as_str().unwrap_or_default().to_owned());
            bodies.push(e.text.as_str());
            created.push(e.created_at);
            updated.push(e.updated_at);
            lineages.push(Json(&e.lineage));
            annotations.push(Json(&e.annotation));
        }
        self.exec(
            "INSERT INTO memory_entries
                (revision_id, ord, entry_id, kind, body, created_at, updated_at, lineage, annotation)
             SELECT $1, * FROM UNNEST($2::int4[], $3::text[], $4::text[], $5::text[],
                                      $6::timestamptz[], $7::timestamptz[], $8::jsonb[], $9::jsonb[])",
            &[
                &rev.revision_id,
                &ords,
                &ids,
                &kinds,
                &bodies,
                &created,
                &updated,
                &lineages,
                &annotations,
            ],
        )
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn revision(&mut self, revision_id: &str) -> Result<Option<Revision>> {
        let rows = self
            .q(
                &format!("SELECT {REVISION_COLUMNS} FROM memory_revisions WHERE revision_id = $1"),
                &[&revision_id],
            )
            .await
            .map_err(db)?;
        Ok(self.load_revisions(rows).await?.into_iter().next())
    }

    async fn revisions(&mut self, memory_id: &str) -> Result<Vec<Revision>> {
        let rows = self
            .q(
                &format!(
                    "SELECT {REVISION_COLUMNS} FROM memory_revisions
                     WHERE memory_id = $1 ORDER BY seq"
                ),
                &[&memory_id],
            )
            .await
            .map_err(db)?;
        self.load_revisions(rows).await
    }

    async fn delete_revision(&mut self, revision_id: &str) -> Result<()> {
        self.exec(
            "DELETE FROM memory_revisions WHERE revision_id = $1",
            &[&revision_id],
        )
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn receipt(
        &mut self,
        tree_id: &str,
        principal: &str,
        key: &str,
    ) -> Result<Option<Receipt>> {
        let row = self
            .q_opt(
                "SELECT record FROM memory_receipts
                 WHERE tree_id = $1 AND principal = $2 AND key = $3",
                &[&tree_id, &principal, &key],
            )
            .await
            .map_err(db)?;
        Ok(row.map(|r| r.get::<_, Json<Receipt>>(0).0))
    }

    async fn put_receipt(&mut self, receipt: &Receipt) -> Result<()> {
        self.exec(
            "INSERT INTO memory_receipts (tree_id, principal, key, memory_id, record)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (tree_id, principal, key) DO UPDATE SET
                    memory_id = EXCLUDED.memory_id, record = EXCLUDED.record",
            &[
                &receipt.tree_id,
                &receipt.principal,
                &receipt.key,
                &receipt.memory_id,
                &Json(receipt),
            ],
        )
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn receipts_for_file(&mut self, memory_id: &str) -> Result<Vec<Receipt>> {
        let rows = self
            .q(
                "SELECT record FROM memory_receipts WHERE memory_id = $1
                 ORDER BY tree_id, principal, key",
                &[&memory_id],
            )
            .await
            .map_err(db)?;
        Ok(rows
            .iter()
            .map(|r| r.get::<_, Json<Receipt>>(0).0)
            .collect())
    }

    async fn commit(mut self: Box<Self>) -> Result<()> {
        let mut conn = self.conn.take().expect("transaction is live");
        let result = conn.client.batch_execute("COMMIT").await;
        // After COMMIT, success or failure, the transaction is over.
        conn.open = false;
        result.map_err(db)
    }
}

/// Erasure journal on Postgres.
pub struct PgErasureJournal {
    client: Mutex<Client>,
    independent: bool,
    url: String,
    schema: Option<String>,
}

impl PgErasureJournal {
    /// `independent` should be true only when this database is backed up and
    /// restored separately from the ledger's.
    pub async fn connect(url: &str, schema: Option<&str>, independent: bool) -> Result<Self> {
        Ok(Self {
            client: Mutex::new(connect(url, schema, JOURNAL_MIGRATIONS).await?),
            independent,
            url: url.into(),
            schema: schema.map(Into::into),
        })
    }

    /// The client, reopened first if it was closed.
    async fn client(&self) -> Result<tokio::sync::MutexGuard<'_, Client>> {
        let mut c = self.client.lock().await;
        if c.is_closed() {
            *c = connect(&self.url, self.schema.as_deref(), JOURNAL_MIGRATIONS).await?;
        }
        Ok(c)
    }
}

#[async_trait]
impl ErasureJournal for PgErasureJournal {
    async fn append(&self, record: ErasureRecord) -> Result<u64> {
        // The stored record's own seq is ignored; the column is authoritative.
        let row = self
            .client()
            .await?
            .query_one(
                "INSERT INTO memory_erasure_journal (tree_id, record) VALUES ($1, $2)
                 RETURNING seq",
                &[&record.tree_id, &Json(&record)],
            )
            .await
            .map_err(db)?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    async fn records(&self, tree_id: &str) -> Result<Vec<ErasureRecord>> {
        let rows = self
            .client()
            .await?
            .query(
                "SELECT seq, record FROM memory_erasure_journal WHERE tree_id = $1 ORDER BY seq",
                &[&tree_id],
            )
            .await
            .map_err(db)?;
        Ok(rows
            .iter()
            .map(|r| {
                let mut record = r.get::<_, Json<ErasureRecord>>(1).0;
                record.seq = r.get::<_, i64>(0) as u64;
                record
            })
            .collect())
    }

    fn independent(&self) -> bool {
        self.independent
    }
}
