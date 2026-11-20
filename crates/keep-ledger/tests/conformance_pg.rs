//! Opt-in tests against a real Postgres.
//! Set KEEP_TEST_POSTGRES_URL (see compose.yaml) to run them. Each test case
//! gets its own `keep_memory_test_*` schema, dropped afterwards.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use keep_core::memory::MemoryLimits;
use keep_ledger::{PgErasureJournal, PgMemoryLedger};
use keep_memory::conformance::{self, Backend, TREE, fixture};
use keep_memory::service::{CreateFile, Edit, ForgetSelector, OpCtx};
use keep_memory::{MemoryError, MemoryService};

fn url() -> Option<String> {
    let url = std::env::var("KEEP_TEST_POSTGRES_URL").ok();
    if url.is_none() {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
    }
    url
}

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// Drops its schema when it goes out of scope, even after a panic.
struct Schema {
    url: String,
    name: String,
}

impl Schema {
    fn new(url: &str) -> Self {
        let name = format!(
            "keep_memory_test_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        );
        // Tag connections so cleanup can close any left inside a transaction.
        let sep = if url.contains('?') { '&' } else { '?' };
        Self {
            url: format!("{url}{sep}application_name={name}"),
            name,
        }
    }

    async fn backend(&self) -> Backend {
        Backend {
            ledger: Arc::new(
                PgMemoryLedger::connect_in_schema(&self.url, Some(&self.name))
                    .await
                    .expect("ledger"),
            ),
            journal: Arc::new(
                PgErasureJournal::connect(&self.url, Some(&self.name), false)
                    .await
                    .expect("journal"),
            ),
        }
    }

    async fn client(&self) -> tokio_postgres::Client {
        let client = keep_postgres::connect(&self.url).await.expect("connect");
        client
            .batch_execute(&format!("SET search_path TO {}", self.name))
            .await
            .expect("search path");
        client
    }
}

impl Drop for Schema {
    fn drop(&mut self) {
        let (url, name) = (self.url.clone(), self.name.clone());
        let _ = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                if let Ok(client) = keep_postgres::connect(&url).await {
                    let _ = client
                        .execute(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
                             WHERE application_name = $1 AND pid <> pg_backend_pid()",
                            &[&name],
                        )
                        .await;
                    let _ = client
                        .batch_execute(&format!(
                            "SET lock_timeout = '10s'; DROP SCHEMA IF EXISTS {name} CASCADE"
                        ))
                        .await;
                }
            });
        })
        .join();
    }
}

#[tokio::test]
async fn postgres_ledger_passes_the_conformance_suite() {
    let Some(url) = url() else { return };
    let schemas = std::sync::Mutex::new(Vec::new());
    conformance::run(|| async {
        let schema = Schema::new(&url);
        let backend = schema.backend().await;
        schemas.lock().unwrap().push(schema);
        backend
    })
    .await;
}

#[tokio::test]
async fn migrations_are_idempotent_and_prefixed() {
    let Some(url) = url() else { return };
    let schema = Schema::new(&url);
    schema.backend().await;
    schema.backend().await;
    let client = schema.client().await;
    let tables: Vec<String> = client
        .query(
            "SELECT table_name::text FROM information_schema.tables WHERE table_schema = $1",
            &[&schema.name],
        )
        .await
        .unwrap()
        .iter()
        .map(|r| r.get(0))
        .collect();
    assert!(tables.len() >= 7, "{tables:?}");
    assert!(
        tables.iter().all(|t| t.starts_with("memory_")),
        "{tables:?}"
    );
    let applied: i64 = client
        .query_one("SELECT count(*) FROM memory_schema_migrations", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(applied, 2);
}

/// Two services on separate connections: the tree lock lets exactly one
/// compare-and-swap win.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_on_separate_connections_have_one_winner() {
    let Some(url) = url() else { return };
    let schema = Schema::new(&url);
    let f = fixture(schema.backend().await, MemoryLimits::default()).await;
    let ctx = OpCtx::new("user:sam");
    let c = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/race.md", "start\n"))
        .await
        .expect("create");
    let mut tasks = Vec::new();
    for i in 0..6 {
        let backend = schema.backend().await;
        let (id, head) = (c.memory_id.clone(), c.head.clone());
        tasks.push(tokio::spawn(async move {
            let svc = MemoryService::new(backend.ledger, backend.journal);
            svc.edit(
                &OpCtx::new("user:sam"),
                TREE,
                &id,
                &head,
                &Edit::Write {
                    text: format!("w{i}\n"),
                },
            )
            .await
        }));
    }
    let mut wins = 0;
    for t in tasks {
        match t.await.expect("join") {
            Ok(_) => wins += 1,
            Err(MemoryError::VersionConflict { .. }) => {}
            Err(e) => panic!("unexpected {e:?}"),
        }
    }
    assert_eq!(wins, 1);
    assert_eq!(f.svc.history(TREE, &c.memory_id).await.unwrap().len(), 2);
}

/// Restore table contents from before a forget, then replay the journal.
#[tokio::test]
async fn journal_replay_cleans_a_restored_backup() {
    let Some(url) = url() else { return };
    let schema = Schema::new(&url);
    let f = fixture(schema.backend().await, MemoryLimits::default()).await;
    let ctx = OpCtx::new("user:sam").with_key("k1");
    let a = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/a.md", "# A\nsecret-xyz\n# B\nfine\n"),
        )
        .await
        .unwrap();
    let client = schema.client().await;
    const TABLES: [&str; 5] = [
        "memory_trees",
        "memory_files",
        "memory_revisions",
        "memory_entries",
        "memory_receipts",
    ];
    for t in TABLES {
        client
            .batch_execute(&format!("CREATE TABLE backup_{t} AS SELECT * FROM {t}"))
            .await
            .unwrap();
    }
    f.svc
        .forget(
            &OpCtx::new("user:sam"),
            TREE,
            &ForgetSelector::Text {
                memory_id: None,
                text: "secret-xyz".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(leaked(&client).await, 0);
    let erasures: i64 = client
        .query_one("SELECT count(*) FROM memory_erasure_journal", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(erasures, 1);
    let journal_text: String = client
        .query_one("SELECT record::text FROM memory_erasure_journal", &[])
        .await
        .unwrap()
        .get(0);
    assert!(!journal_text.contains("secret-xyz"));

    // Restore the backup.
    client
        .batch_execute(
            "TRUNCATE memory_entries, memory_revisions, memory_receipts, memory_files, memory_trees",
        )
        .await
        .unwrap();
    for t in TABLES {
        client
            .batch_execute(&format!("INSERT INTO {t} SELECT * FROM backup_{t}"))
            .await
            .unwrap();
    }
    assert!(leaked(&client).await > 0);
    f.svc.replay_erasure_journal(TREE).await.unwrap();
    assert_eq!(leaked(&client).await, 0);
    assert!(matches!(
        f.svc.view(TREE, &a.memory_id).await,
        Err(MemoryError::NotFound { .. })
    ));
    assert_eq!(
        f.svc
            .create(
                &ctx,
                TREE,
                &CreateFile::new("/topics/a.md", "# A\nsecret-xyz\n# B\nfine\n")
            )
            .await,
        Err(MemoryError::IdempotencyKeyRetired)
    );
}

/// Rows anywhere in the ledger that still hold the forgotten text.
async fn leaked(client: &tokio_postgres::Client) -> i64 {
    client
        .query_one(
            "SELECT (SELECT count(*) FROM memory_entries WHERE body LIKE '%secret-xyz%')
                  + (SELECT count(*) FROM memory_files WHERE record::text LIKE '%secret-xyz%')
                  + (SELECT count(*) FROM memory_receipts WHERE record::text LIKE '%secret-xyz%')",
            &[],
        )
        .await
        .unwrap()
        .get(0)
}
