//! Memory (`keep-ledger`) and Knowledge (`keep-knowledge`) can share one
//! Postgres schema: their tables and migration records are disjoint, so
//! either may migrate first and reconnecting is a no-op. Opt-in through
//! `KEEP_TEST_POSTGRES_URL`.

use keep_knowledge::PgLedger;
use keep_ledger::{PgErasureJournal, PgMemoryLedger};
use tokio_postgres::{Client, NoTls};

async fn admin(url: &str) -> Client {
    let (client, conn) = tokio_postgres::connect(url, NoTls).await.unwrap();
    tokio::spawn(conn);
    client
}

fn in_schema(url: &str, schema: &str) -> String {
    let sep = if url.contains('?') { '&' } else { '?' };
    format!("{url}{sep}options=-csearch_path%3D{schema},public")
}

async fn memory(url: &str, schema: &str) {
    PgMemoryLedger::connect_in_schema(url, Some(schema))
        .await
        .unwrap();
    PgErasureJournal::connect(url, Some(schema), false)
        .await
        .unwrap();
}

async fn knowledge(url: &str, schema: &str) {
    PgLedger::connect(&in_schema(url, schema)).await.unwrap();
}

async fn versions(client: &Client, schema: &str, table: &str) -> Vec<String> {
    client
        .query(
            &format!("SELECT version FROM {schema}.{table} ORDER BY version"),
            &[],
        )
        .await
        .unwrap()
        .iter()
        .map(|r| r.get(0))
        .collect()
}

#[tokio::test]
async fn memory_and_knowledge_migrate_into_one_schema_in_either_order() {
    let Ok(url) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let client = admin(&url).await;
    let pid = std::process::id();
    for (order, memory_first) in [("mk", true), ("km", false)] {
        let schema = format!("keep_shared_test_{order}_{pid}");
        client
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema};"
            ))
            .await
            .unwrap();
        let result = async {
            for _ in 0..2 {
                if memory_first {
                    memory(&url, &schema).await;
                    knowledge(&url, &schema).await;
                } else {
                    knowledge(&url, &schema).await;
                    memory(&url, &schema).await;
                }
            }
            let tables: Vec<String> = client
                .query(
                    "SELECT table_name::text FROM information_schema.tables WHERE table_schema = $1",
                    &[&schema],
                )
                .await
                .unwrap()
                .iter()
                .map(|r| r.get(0))
                .collect();
            (
                tables,
                versions(&client, &schema, "memory_schema_migrations").await,
                versions(&client, &schema, "knowledge_schema_migrations").await,
            )
        }
        .await;
        client
            .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
            .await
            .unwrap();
        let (tables, mem, know) = result;
        assert!(
            tables
                .iter()
                .all(|t| t.starts_with("memory_") || t.starts_with("knowledge_")),
            "{tables:?}"
        );
        assert!(tables.iter().any(|t| t == "memory_trees"));
        assert!(tables.iter().any(|t| t == "knowledge_sources"));
        assert_eq!(mem, ["memory_0001_ledger", "memory_0002_erasure_journal"]);
        assert_eq!(know, ["knowledge_0001_ledger"]);
    }
}
