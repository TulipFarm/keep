//! Per-run storage. An executed run against Postgres writes into its own
//! schema, `keepbench_<run>`, and against Qdrant into its own collections,
//! `keepbench_<run>` (dense) and `keepbench_k_<run>` (Knowledge). It drops
//! them when it ends, so concurrent and interrupted runs never share or
//! leave rows in the database's default schema. A Postgres URL that already
//! sets `options` (for example its own `search_path`) is used as given.

use crate::manifest::{LoadedManifest, StorageProfile};
use crate::runner::Mode;

pub struct RunStorage {
    pub profile: StorageProfile,
    pub postgres_url: Option<String>,
    pub qdrant_url: Option<String>,
    pub qdrant_api_key: Option<String>,
    pub dense_collection: String,
    pub knowledge_collection: String,
    /// Admin URL and schema, when this run created the schema.
    schema: Option<(String, String)>,
}

/// Lowercase letters, digits and underscores, safe as a Postgres schema
/// and a Qdrant collection name.
pub fn run_tag(run_id: &str) -> String {
    run_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// `url` with `search_path` set to `schema`, then `public` for extension
/// types such as pgvector's.
pub fn with_search_path(url: &str, schema: &str) -> String {
    let sep = if url.contains('?') { '&' } else { '?' };
    format!("{url}{sep}options=-csearch_path%3D{schema},public")
}

impl RunStorage {
    /// Resolve the manifest's storage for one run, creating its schema when
    /// the run executes against Postgres.
    pub async fn open(lm: &LoadedManifest, run_id: &str, mode: Mode) -> anyhow::Result<Self> {
        let s = &lm.manifest.storage;
        let tag = run_tag(run_id);
        let mut out = Self {
            profile: s.profile,
            postgres_url: None,
            qdrant_url: None,
            qdrant_api_key: None,
            dense_collection: format!("keepbench_{tag}"),
            knowledge_collection: format!("keepbench_k_{tag}"),
            schema: None,
        };
        if mode == Mode::Estimate || s.profile == StorageProfile::InMemory {
            out.profile = StorageProfile::InMemory;
            return Ok(out);
        }
        let env = |name: &str| {
            std::env::var(name)
                .map_err(|_| anyhow::anyhow!("storage profile needs environment variable {name}"))
        };
        let admin = env(&s.postgres_url_env)?;
        if s.profile == StorageProfile::PostgresQdrant {
            out.qdrant_url = Some(env(&s.qdrant_url_env)?);
            out.qdrant_api_key = s
                .qdrant_api_key_env
                .as_deref()
                .and_then(|k| std::env::var(k).ok());
        }
        if admin.contains("options=") {
            out.postgres_url = Some(admin);
            return Ok(out);
        }
        let schema = format!("keepbench_{tag}");
        let (client, conn) = tokio_postgres::connect(&admin, tokio_postgres::NoTls).await?;
        let conn = tokio::spawn(conn);
        client
            .batch_execute(&format!("CREATE SCHEMA IF NOT EXISTS {schema}"))
            .await?;
        drop(client);
        let _ = conn.await;
        out.postgres_url = Some(with_search_path(&admin, &schema));
        out.schema = Some((admin, schema));
        Ok(out)
    }

    /// The schema this run created, if any.
    pub fn schema(&self) -> Option<&str> {
        self.schema.as_ref().map(|(_, s)| s.as_str())
    }

    /// Whether [`Self::close`] removes everything the run stored, so
    /// deleting its corpora and namespaces one by one is unnecessary.
    pub fn drops_everything(&self) -> bool {
        match self.profile {
            StorageProfile::InMemory => true,
            _ => self.schema.is_some(),
        }
    }

    /// Drop the run's schema and Qdrant collections. Returns warnings.
    pub async fn close(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let Some(url) = &self.qdrant_url {
            let q = keep_qdrant::QdrantIndex::new(url, self.qdrant_api_key.clone());
            for c in [&self.dense_collection, &self.knowledge_collection] {
                if let Err(e) = q.delete_collection(c).await {
                    warnings.push(format!("could not delete Qdrant collection {c}: {e}"));
                }
            }
        }
        if let Some((admin, schema)) = &self.schema {
            let dropped = async {
                let (client, conn) = tokio_postgres::connect(admin, tokio_postgres::NoTls).await?;
                tokio::spawn(conn);
                client
                    .batch_execute(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE"))
                    .await
            }
            .await;
            if let Err(e) = dropped {
                warnings.push(format!("could not drop Postgres schema {schema}: {e}"));
            }
        }
        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_tags_are_safe_names() {
        let tag = run_tag("20261113T062810.353Z-c8c628390732");
        assert_eq!(tag, "20261113t062810_353z_c8c628390732");
        assert_eq!(
            with_search_path("postgres://h/db", "keepbench_x"),
            "postgres://h/db?options=-csearch_path%3Dkeepbench_x,public"
        );
        assert_eq!(
            with_search_path("postgres://h/db?sslmode=disable", "s"),
            "postgres://h/db?sslmode=disable&options=-csearch_path%3Ds,public"
        );
    }
}
