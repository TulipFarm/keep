//! Postgres ledger. Tables are created by the migrations in `migrations/`
//! (all prefixed `knowledge_`), applied once per schema on connect.
//!
//! Connections go through [`keep_postgres::connect`], which requires TLS for
//! remote hosts. The ledger holds a single client, reopened if it closes; a
//! pool is still to come.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use keep_store::ScoredId;
use tokio::sync::Mutex;
use tokio_postgres::{Client, GenericClient, Row, Transaction};

use crate::ledger::{
    ApplyOutcome, ChangeSet, EntityStat, IndexInputs, IngestBatch, IngestReport, KnowledgeError,
    Ledger, MAX_JOB_ATTEMPTS, Projection, PurgeReport, Result, SourceStat, episode_set,
    normalize_name, retry_delay, same_source,
};
use crate::memory_ledger::{LEASE_SECONDS, splice_versions};
use crate::model::{
    Chunk, Entity, Episode, ErasureEntry, Evidence, Fact, FactStatus, FactVersion, Job, JobCounts,
    JobKind, JobState, NewJob, Source, SourceKind, Time, TimeBasis, digest_hex,
};

const MIGRATIONS: &[(&str, &str)] = &[
    (
        "knowledge_0001_ledger",
        include_str!("../migrations/knowledge_0001_ledger.sql"),
    ),
    (
        "knowledge_0002_source_metadata",
        include_str!("../migrations/knowledge_0002_source_metadata.sql"),
    ),
];

pub struct PgLedger {
    url: String,
    client: Mutex<Client>,
}

fn db(e: impl std::fmt::Display) -> KnowledgeError {
    KnowledgeError::Storage(e.to_string())
}

impl PgLedger {
    /// Connect and apply pending migrations. The schema is whatever the
    /// connection's `search_path` selects (e.g. `?options=-csearch_path%3Dx`).
    pub async fn connect(url: &str) -> Result<Self> {
        let mut client = open(url).await?;
        migrate(&mut client).await?;
        Ok(Self {
            url: url.into(),
            client: Mutex::new(client),
        })
    }

    /// The client, reopened first if the server or network closed it
    /// (for example after a long idle period).
    async fn client(&self) -> Result<tokio::sync::MutexGuard<'_, Client>> {
        let mut c = self.client.lock().await;
        if c.is_closed() {
            *c = open(&self.url).await?;
        }
        Ok(c)
    }
}

async fn open(url: &str) -> Result<Client> {
    keep_postgres::connect(url).await.map_err(db)
}

async fn migrate(client: &mut Client) -> Result<()> {
    let tx = client.transaction().await.map_err(db)?;
    tx.batch_execute(
        "SELECT pg_advisory_xact_lock(hashtext('keep_knowledge_migrations'));
         CREATE TABLE IF NOT EXISTS knowledge_schema_migrations (
             version text PRIMARY KEY,
             applied_at timestamptz NOT NULL DEFAULT now()
         );",
    )
    .await
    .map_err(db)?;
    for (version, sql) in MIGRATIONS {
        let done = tx
            .query_opt(
                "SELECT 1 FROM knowledge_schema_migrations WHERE version = $1",
                &[version],
            )
            .await
            .map_err(db)?
            .is_some();
        if !done {
            tx.batch_execute(sql).await.map_err(db)?;
            tx.execute(
                "INSERT INTO knowledge_schema_migrations (version) VALUES ($1)",
                &[version],
            )
            .await
            .map_err(db)?;
        }
    }
    tx.commit().await.map_err(db)
}

fn source_row(r: &Row) -> Source {
    Source {
        corpus: r.get("corpus"),
        id: r.get("id"),
        partition: r.get("partition_label"),
        collection: r.get("collection"),
        title: r.get("title"),
        kind: SourceKind::parse(r.get("kind")),
        created_at: r.get("created_at"),
        metadata: serde_json::from_value(r.get("metadata")).unwrap_or_default(),
    }
}

fn episode_row(r: &Row) -> Episode {
    Episode {
        corpus: r.get("corpus"),
        id: r.get("id"),
        source_id: r.get("source_id"),
        partition: r.get("partition_label"),
        ordinal: r.get("ordinal"),
        speaker: r.get("speaker"),
        occurred_at: r.get("occurred_at"),
        recorded_at: r.get("recorded_at"),
        text: r.get("text"),
        content_hash: r.get("content_hash"),
    }
}

fn chunk_row(r: &Row) -> Chunk {
    Chunk {
        corpus: r.get("corpus"),
        id: r.get("id"),
        episode_id: r.get("episode_id"),
        source_id: r.get("source_id"),
        partition: r.get("partition_label"),
        ordinal: r.get("ordinal"),
        char_start: r.get("char_start"),
        char_end: r.get("char_end"),
        speaker: r.get("speaker"),
        text: r.get("text"),
        occurred_at: r.get("occurred_at"),
        recorded_at: r.get("recorded_at"),
    }
}

fn version_row(r: &Row) -> FactVersion {
    FactVersion {
        version: r.get("version"),
        valid_from: r.get("valid_from"),
        valid_to: r.get("valid_to"),
        valid_from_basis: TimeBasis::parse(r.get("valid_from_basis")),
        status: FactStatus::parse(r.get("status")),
        recorded_at: r.get("recorded_at"),
        invalidated_at: r.get("invalidated_at"),
        cause: r.get("cause"),
    }
}

fn evidence_row(r: &Row) -> Evidence {
    Evidence {
        corpus: r.get("corpus"),
        id: r.get("id"),
        fact_id: r.get("fact_id"),
        episode_id: r.get("episode_id"),
        source_id: r.get("source_id"),
        partition: r.get("partition_label"),
        char_start: r.get("char_start"),
        char_end: r.get("char_end"),
        quote: r.get("quote"),
        extractor: r.get("extractor"),
        recorded_at: r.get("recorded_at"),
    }
}

fn job_row(r: &Row) -> Job {
    Job {
        id: r.get("id"),
        corpus: r.get("corpus"),
        kind: JobKind::parse(r.get("kind")),
        target: r.get("target"),
        state: JobState::parse(r.get("state")),
        attempts: r.get("attempts"),
        max_attempts: r.get("max_attempts"),
        next_attempt_at: r.get("next_attempt_at"),
        lease: r.get("lease"),
        last_error: r.get("last_error"),
    }
}

async fn load_facts<C: GenericClient>(c: &C, corpus: &str, ids: &[String]) -> Result<Vec<Fact>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let rows = c
        .query(
            "SELECT * FROM knowledge_facts WHERE corpus = $1 AND id = ANY($2)",
            &[&corpus, &ids],
        )
        .await
        .map_err(db)?;
    let mut facts: BTreeMap<String, Fact> = rows
        .iter()
        .map(|r| {
            let f = Fact {
                corpus: r.get("corpus"),
                id: r.get("id"),
                partition: r.get("partition_label"),
                subject: r.get("subject"),
                predicate: r.get("predicate"),
                object: r.get("object"),
                object_entity: r.get("object_entity"),
                text: r.get("text"),
                versions: vec![],
                evidence: vec![],
            };
            (f.id.clone(), f)
        })
        .collect();
    for r in c
        .query(
            "SELECT * FROM knowledge_fact_versions WHERE corpus = $1 AND fact_id = ANY($2)
             ORDER BY fact_id, version",
            &[&corpus, &ids],
        )
        .await
        .map_err(db)?
    {
        if let Some(f) = facts.get_mut(r.get::<_, &str>("fact_id")) {
            f.versions.push(version_row(&r));
        }
    }
    for r in c
        .query(
            "SELECT * FROM knowledge_evidence WHERE corpus = $1 AND fact_id = ANY($2)
             ORDER BY fact_id, recorded_at, id",
            &[&corpus, &ids],
        )
        .await
        .map_err(db)?
    {
        if let Some(f) = facts.get_mut(r.get::<_, &str>("fact_id")) {
            f.evidence.push(evidence_row(&r));
        }
    }
    // Preserve the caller's order.
    Ok(ids.iter().filter_map(|id| facts.remove(id)).collect())
}

async fn load_entity<C: GenericClient>(
    c: &C,
    corpus: &str,
    id: &str,
    sources: Option<&[String]>,
) -> Result<Option<Entity>> {
    let Some(r) = c
        .query_opt(
            "SELECT * FROM knowledge_entities WHERE corpus = $1 AND id = $2",
            &[&corpus, &id],
        )
        .await
        .map_err(db)?
    else {
        return Ok(None);
    };
    let aliases = c
        .query(
            "SELECT DISTINCT alias FROM knowledge_entity_aliases
             WHERE corpus = $1 AND entity_id = $2 AND ($3::text[] IS NULL OR source_id = ANY($3))",
            &[&corpus, &id, &sources],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get::<_, String>(0))
        .collect();
    Ok(Some(Entity {
        corpus: r.get("corpus"),
        id: r.get("id"),
        partition: r.get("partition_label"),
        name: r.get("name"),
        kind: r.get("kind"),
        aliases,
    }))
}

async fn bump(tx: &Transaction<'_>, corpus: &str, partition: &str) -> Result<()> {
    tx.execute(
        "INSERT INTO knowledge_partitions (corpus, partition_label, generation) VALUES ($1, $2, 1)
         ON CONFLICT (corpus, partition_label)
         DO UPDATE SET generation = knowledge_partitions.generation + 1",
        &[&corpus, &partition],
    )
    .await
    .map_err(db)?;
    Ok(())
}

async fn enqueue_tx<C: GenericClient>(
    c: &C,
    corpus: &str,
    jobs: &[NewJob],
    now: Time,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let kinds: Vec<&str> = jobs.iter().map(|j| j.kind.as_str()).collect();
    let targets: Vec<&str> = jobs.iter().map(|j| j.target.as_str()).collect();
    // One statement; ids follow the input order, as they did row by row.
    c.execute(
        "INSERT INTO knowledge_jobs (corpus, kind, target, state, attempts, max_attempts, next_attempt_at)
         SELECT $1, j.kind, j.target, 'pending', 0, $4, $5
         FROM unnest($2::text[], $3::text[]) WITH ORDINALITY AS j(kind, target, n)
         ORDER BY j.n",
        &[&corpus, &kinds, &targets, &MAX_JOB_ATTEMPTS, &now],
    )
    .await
    .map_err(db)?;
    Ok(())
}

async fn insert_version<C: GenericClient>(
    c: &C,
    corpus: &str,
    fact_id: &str,
    v: &FactVersion,
) -> Result<()> {
    c.execute(
        "INSERT INTO knowledge_fact_versions
         (corpus, fact_id, version, valid_from, valid_to, valid_from_basis, status, recorded_at, invalidated_at, cause)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        &[
            &corpus,
            &fact_id,
            &v.version,
            &v.valid_from,
            &v.valid_to,
            &v.valid_from_basis.as_str(),
            &v.status.as_str(),
            &v.recorded_at,
            &v.invalidated_at,
            &v.cause,
        ],
    )
    .await
    .map_err(db)?;
    Ok(())
}

async fn insert_evidence<C: GenericClient>(c: &C, e: &Evidence) -> Result<()> {
    c.execute(
        "INSERT INTO knowledge_evidence
         (corpus, id, fact_id, episode_id, source_id, partition_label, char_start, char_end, quote, extractor, recorded_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
         ON CONFLICT DO NOTHING",
        &[
            &e.corpus,
            &e.id,
            &e.fact_id,
            &e.episode_id,
            &e.source_id,
            &e.partition,
            &e.char_start,
            &e.char_end,
            &e.quote,
            &e.extractor,
            &e.recorded_at,
        ],
    )
    .await
    .map_err(db)?;
    Ok(())
}

/// `a | b | c` from the query's alphanumeric tokens, safe for `to_tsquery`.
fn tsquery(query: &str) -> String {
    let mut seen = BTreeSet::new();
    keep_store::lexical::tokenize(query)
        .into_iter()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|t| !t.is_empty() && seen.insert(t.clone()))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[async_trait]
impl Ledger for PgLedger {
    fn backend(&self) -> &'static str {
        "postgres"
    }

    async fn ingest(&self, batch: IngestBatch) -> Result<IngestReport> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let src = &batch.source;
        let corpus = src.corpus.as_str();
        let mut report = IngestReport::default();
        let mut old = tx
            .query_opt(
                "SELECT * FROM knowledge_sources WHERE corpus = $1 AND id = $2 FOR UPDATE",
                &[&corpus, &src.id],
            )
            .await
            .map_err(db)?
            .map(|r| source_row(&r));
        if let Some(prev) = &old {
            let same_meta = same_source(prev, src);
            let changed = batch.replace
                && (!same_meta
                    || {
                        let held: BTreeSet<(String, String)> = tx
                    .query(
                        "SELECT id, content_hash FROM knowledge_episodes WHERE corpus = $1 AND source_id = $2",
                        &[&corpus, &src.id],
                    )
                    .await
                    .map_err(db)?
                    .iter()
                    .map(|r| (r.get(0), r.get(1)))
                    .collect();
                        held != episode_set(&batch.episodes)
                    });
            if changed {
                let purged = purge_tx(&tx, corpus, prev, batch.now).await?;
                report.replaced = true;
                report.purged = Some(purged);
                old = None;
            } else if !same_meta {
                return Err(KnowledgeError::Conflict(
                    "source already exists with different metadata".into(),
                ));
            }
        }
        if old.is_none() {
            tx.execute(
                "INSERT INTO knowledge_sources (corpus, id, partition_label, collection, title, kind, created_at, metadata)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                &[
                    &corpus,
                    &src.id,
                    &src.partition,
                    &src.collection,
                    &src.title,
                    &src.kind.as_str(),
                    &src.created_at,
                    &serde_json::to_value(&src.metadata).map_err(db)?,
                ],
            )
            .await
            .map_err(db)?;
            report.source_created = !report.replaced;
        }
        let ids: Vec<String> = batch.episodes.iter().map(|e| e.id.clone()).collect();
        let existing: BTreeMap<String, (String, String)> = tx
            .query(
                "SELECT id, content_hash, source_id FROM knowledge_episodes WHERE corpus = $1 AND id = ANY($2)",
                &[&corpus, &ids],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| (r.get(0), (r.get(1), r.get(2))))
            .collect();
        let mut fresh = BTreeSet::new();
        for e in &batch.episodes {
            match existing.get(&e.id) {
                Some((hash, sid)) if *hash == e.content_hash && *sid == e.source_id => {
                    report.episodes_unchanged += 1
                }
                Some(_) => {
                    return Err(KnowledgeError::Conflict(
                        "episode already exists with different content".into(),
                    ));
                }
                None => {
                    fresh.insert(e.id.clone());
                }
            }
        }
        let eps: Vec<&Episode> = batch
            .episodes
            .iter()
            .filter(|e| fresh.contains(&e.id))
            .collect();
        let chs: Vec<&Chunk> = batch
            .chunks
            .iter()
            .filter(|c| fresh.contains(&c.episode_id))
            .collect();
        if !eps.is_empty() {
            tx.execute(
                "INSERT INTO knowledge_episodes
                 (corpus, id, source_id, partition_label, ordinal, speaker, occurred_at, recorded_at, text, content_hash)
                 SELECT $1, e.* FROM unnest($2::text[], $3::text[], $4::text[], $5::int4[], $6::text[],
                                            $7::timestamptz[], $8::timestamptz[], $9::text[], $10::text[]) AS e",
                &[
                    &corpus,
                    &eps.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.source_id.as_str()).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.partition.as_str()).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.ordinal).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.speaker.as_str()).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.occurred_at).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.recorded_at).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(),
                    &eps.iter().map(|e| e.content_hash.as_str()).collect::<Vec<_>>(),
                ],
            )
            .await
            .map_err(db)?;
        }
        if !chs.is_empty() {
            tx.execute(
                "INSERT INTO knowledge_chunks
                 (corpus, id, episode_id, source_id, partition_label, ordinal, char_start, char_end, speaker, text, occurred_at, recorded_at)
                 SELECT $1, c.* FROM unnest($2::text[], $3::text[], $4::text[], $5::text[], $6::int4[], $7::int4[],
                                            $8::int4[], $9::text[], $10::text[], $11::timestamptz[], $12::timestamptz[]) AS c",
                &[
                    &corpus,
                    &chs.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.episode_id.as_str()).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.source_id.as_str()).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.partition.as_str()).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.ordinal).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.char_start).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.char_end).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.speaker.as_str()).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.occurred_at).collect::<Vec<_>>(),
                    &chs.iter().map(|c| c.recorded_at).collect::<Vec<_>>(),
                ],
            )
            .await
            .map_err(db)?;
        }
        report.episodes_added = eps.len();
        report.chunks_added = chs.len();
        let jobs: Vec<NewJob> = eps
            .iter()
            .map(|e| NewJob {
                kind: JobKind::Extract,
                target: e.id.clone(),
            })
            .chain(chs.iter().map(|c| NewJob {
                kind: JobKind::Project,
                target: c.id.clone(),
            }))
            .collect();
        report.jobs_enqueued = jobs.len();
        enqueue_tx(&tx, corpus, &jobs, batch.now).await?;
        if report.source_created || report.replaced || report.episodes_added > 0 {
            bump(&tx, corpus, &src.partition).await?;
        }
        tx.commit().await.map_err(db)?;
        Ok(report)
    }

    async fn source(&self, corpus: &str, id: &str) -> Result<Option<Source>> {
        let c = self.client().await?;
        Ok(c.query_opt(
            "SELECT * FROM knowledge_sources WHERE corpus = $1 AND id = $2",
            &[&corpus, &id],
        )
        .await
        .map_err(db)?
        .map(|r| source_row(&r)))
    }

    async fn episode(&self, corpus: &str, id: &str) -> Result<Option<Episode>> {
        let c = self.client().await?;
        Ok(c.query_opt(
            "SELECT * FROM knowledge_episodes WHERE corpus = $1 AND id = $2",
            &[&corpus, &id],
        )
        .await
        .map_err(db)?
        .map(|r| episode_row(&r)))
    }

    async fn source_episodes(&self, corpus: &str, source_id: &str) -> Result<Vec<Episode>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT * FROM knowledge_episodes WHERE corpus = $1 AND source_id = $2 \
             ORDER BY ordinal",
            &[&corpus, &source_id],
        )
        .await
        .map_err(db)?
        .iter()
        .map(episode_row)
        .collect())
    }

    async fn chunks(&self, corpus: &str, ids: &[String]) -> Result<Vec<Chunk>> {
        let c = self.client().await?;
        let mut by_id: BTreeMap<String, Chunk> = c
            .query(
                "SELECT * FROM knowledge_chunks WHERE corpus = $1 AND id = ANY($2)",
                &[&corpus, &ids],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| {
                let ch = chunk_row(r);
                (ch.id.clone(), ch)
            })
            .collect();
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    async fn facts(&self, corpus: &str, ids: &[String]) -> Result<Vec<Fact>> {
        let c = self.client().await?;
        load_facts(&*c, corpus, ids).await
    }

    async fn evidence(&self, corpus: &str, id: &str) -> Result<Option<Evidence>> {
        let c = self.client().await?;
        Ok(c.query_opt(
            "SELECT * FROM knowledge_evidence WHERE corpus = $1 AND id = $2",
            &[&corpus, &id],
        )
        .await
        .map_err(db)?
        .map(|r| evidence_row(&r)))
    }

    async fn entity(&self, corpus: &str, id: &str) -> Result<Option<Entity>> {
        let c = self.client().await?;
        load_entity(&*c, corpus, id, None).await
    }

    async fn find_entity(
        &self,
        corpus: &str,
        partition: &str,
        normalized: &str,
    ) -> Result<Option<Entity>> {
        let c = self.client().await?;
        let id: Option<String> = c
            .query_opt(
                "SELECT en.id FROM knowledge_entities en
                 WHERE en.corpus = $1 AND en.partition_label = $2
                   AND (en.norm_name = $3 OR EXISTS (
                     SELECT 1 FROM knowledge_entity_aliases a
                     WHERE a.corpus = en.corpus AND a.entity_id = en.id AND a.norm_alias = $3))
                 ORDER BY en.id LIMIT 1",
                &[&corpus, &partition, &normalized],
            )
            .await
            .map_err(db)?
            .map(|r| r.get(0));
        match id {
            Some(id) => load_entity(&*c, corpus, &id, None).await,
            None => Ok(None),
        }
    }

    async fn entities_named(
        &self,
        corpus: &str,
        partitions: &[String],
        normalized: &str,
        sources: Option<&[String]>,
    ) -> Result<Vec<Entity>> {
        let c = self.client().await?;
        let ids: Vec<String> = c
            .query(
                "SELECT en.id FROM knowledge_entities en
                 WHERE en.corpus = $1 AND en.partition_label = ANY($2)
                   AND (en.norm_name = $3 OR EXISTS (
                     SELECT 1 FROM knowledge_entity_aliases a
                     WHERE a.corpus = en.corpus AND a.entity_id = en.id AND a.norm_alias = $3
                       AND ($4::text[] IS NULL OR a.source_id = ANY($4))))
                 ORDER BY en.id",
                &[&corpus, &partitions, &normalized, &sources],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| r.get(0))
            .collect();
        let mut out = Vec::new();
        for id in ids {
            if let Some(e) = load_entity(&*c, corpus, &id, sources).await? {
                out.push(e);
            }
        }
        Ok(out)
    }

    async fn aliases_from(
        &self,
        corpus: &str,
        entity_id: &str,
        sources: &[String],
    ) -> Result<BTreeSet<String>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT DISTINCT alias FROM knowledge_entity_aliases
             WHERE corpus = $1 AND entity_id = $2 AND source_id = ANY($3)",
            &[&corpus, &entity_id, &sources],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect())
    }

    async fn current_facts(
        &self,
        corpus: &str,
        partition: &str,
        subject: &str,
        predicate: &str,
    ) -> Result<Vec<Fact>> {
        let c = self.client().await?;
        let ids: Vec<String> = c
            .query(
                "SELECT f.id FROM knowledge_facts f
                 WHERE f.corpus = $1 AND f.partition_label = $2 AND f.subject = $3 AND f.predicate = $4
                   AND EXISTS (SELECT 1 FROM knowledge_fact_versions v
                               WHERE v.corpus = f.corpus AND v.fact_id = f.id AND v.invalidated_at IS NULL)
                 ORDER BY f.id",
                &[&corpus, &partition, &subject, &predicate],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| r.get(0))
            .collect();
        load_facts(&*c, corpus, &ids).await
    }

    async fn facts_about(
        &self,
        corpus: &str,
        partitions: &[String],
        entity_id: &str,
        limit: usize,
        sources: Option<&[String]>,
    ) -> Result<Vec<Fact>> {
        let c = self.client().await?;
        let ids: Vec<String> = c
            .query(
                "SELECT f.id FROM knowledge_facts f
                 WHERE f.corpus = $1 AND f.partition_label = ANY($2)
                   AND (f.subject = $3 OR f.object_entity = $3)
                   AND ($5::text[] IS NULL OR EXISTS (SELECT 1 FROM knowledge_evidence ev WHERE ev.corpus = f.corpus AND ev.fact_id = f.id AND ev.source_id = ANY($5)))
                 ORDER BY GREATEST(
                   (SELECT max(recorded_at) FROM knowledge_fact_versions v WHERE v.corpus = f.corpus AND v.fact_id = f.id),
                   (SELECT max(recorded_at) FROM knowledge_evidence e WHERE e.corpus = f.corpus AND e.fact_id = f.id)
                 ) DESC NULLS LAST, f.id
                 LIMIT $4",
                &[&corpus, &partitions, &entity_id, &(limit as i64), &sources],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| r.get(0))
            .collect();
        load_facts(&*c, corpus, &ids).await
    }

    async fn apply(&self, cs: ChangeSet) -> Result<ApplyOutcome> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let corpus = cs.corpus.as_str();
        // Holding a share lock on the cited episodes makes a concurrent
        // purge or replacement wait for this transaction, then delete what
        // it wrote. Content hashes catch a replacement that already ran.
        let mut ids: Vec<&str> = vec![cs.episode_id.as_str()];
        ids.extend(cs.episodes.iter().map(|(id, _)| id.as_str()));
        let found: BTreeMap<String, String> = tx
            .query(
                "SELECT id, content_hash FROM knowledge_episodes
                 WHERE corpus = $1 AND id = ANY($2) ORDER BY id FOR SHARE",
                &[&corpus, &ids],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| (r.get(0), r.get(1)))
            .collect();
        if !found.contains_key(&cs.episode_id)
            || !cs
                .episodes
                .iter()
                .all(|(id, hash)| found.get(id) == Some(hash))
        {
            return Ok(ApplyOutcome::EpisodeGone);
        }
        for v in &cs.versions {
            let cur: Option<i32> = tx
                .query_opt(
                    "SELECT version FROM knowledge_fact_versions
                     WHERE corpus = $1 AND fact_id = $2 AND invalidated_at IS NULL FOR UPDATE",
                    &[&corpus, &v.fact_id],
                )
                .await
                .map_err(db)?
                .map(|r| r.get(0));
            if cur != Some(v.expected_current) {
                return Ok(ApplyOutcome::Conflict);
            }
        }
        for e in &cs.entities {
            tx.execute(
                "INSERT INTO knowledge_entities (corpus, id, partition_label, name, norm_name, kind)
                 VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
                &[&corpus, &e.id, &cs.partition, &e.name, &normalize_name(&e.name), &e.kind],
            )
            .await
            .map_err(db)?;
        }
        for a in &cs.aliases {
            tx.execute(
                "INSERT INTO knowledge_entity_aliases (corpus, entity_id, alias, norm_alias, source_id)
                 VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
                &[&corpus, &a.entity_id, &a.alias, &normalize_name(&a.alias), &a.source_id],
            )
            .await
            .map_err(db)?;
        }
        for f in &cs.new_facts {
            let n = tx
                .execute(
                    "INSERT INTO knowledge_facts (corpus, id, partition_label, subject, predicate, object, object_entity, text)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT DO NOTHING",
                    &[
                        &corpus,
                        &f.id,
                        &f.partition,
                        &f.subject,
                        &f.predicate,
                        &f.object,
                        &f.object_entity,
                        &f.text,
                    ],
                )
                .await
                .map_err(db)?;
            if n == 0 {
                return Ok(ApplyOutcome::Conflict);
            }
            for v in &f.versions {
                insert_version(&tx, corpus, &f.id, v).await?;
            }
            for e in &f.evidence {
                insert_evidence(&tx, e).await?;
            }
        }
        for e in &cs.evidence {
            let exists = tx
                .query_opt(
                    "SELECT 1 FROM knowledge_facts WHERE corpus = $1 AND id = $2",
                    &[&corpus, &e.fact_id],
                )
                .await
                .map_err(db)?
                .is_some();
            if !exists {
                return Ok(ApplyOutcome::Conflict);
            }
            insert_evidence(&tx, e).await?;
        }
        for v in &cs.versions {
            tx.execute(
                "UPDATE knowledge_fact_versions SET invalidated_at = $3
                 WHERE corpus = $1 AND fact_id = $2 AND invalidated_at IS NULL",
                &[&corpus, &v.fact_id, &v.version.recorded_at],
            )
            .await
            .map_err(db)?;
            insert_version(&tx, corpus, &v.fact_id, &v.version).await?;
        }
        enqueue_tx(&tx, corpus, &cs.jobs, cs.now).await?;
        if !cs.is_empty() {
            bump(&tx, corpus, &cs.partition).await?;
        }
        tx.commit().await.map_err(db)?;
        Ok(ApplyOutcome::Applied)
    }

    async fn lexical(
        &self,
        corpus: &str,
        partitions: &[String],
        query: &str,
        limit: usize,
        sources: Option<&[String]>,
    ) -> Result<Vec<ScoredId>> {
        let q = tsquery(query);
        if q.is_empty() {
            return Ok(vec![]);
        }
        let c = self.client().await?;
        let rows = c
            .query(
                "WITH q AS (SELECT to_tsquery('english', $3) AS q)
                 SELECT id, score FROM (
                   SELECT ch.id, ts_rank_cd(ch.tsv, q.q)::real AS score
                   FROM knowledge_chunks ch, q
                   WHERE ch.corpus = $1 AND ch.partition_label = ANY($2) AND ch.tsv @@ q.q
                     AND ($5::text[] IS NULL OR ch.source_id = ANY($5))
                   UNION ALL
                   SELECT f.id, ts_rank_cd(f.tsv, q.q)::real AS score
                   FROM knowledge_facts f, q
                   WHERE f.corpus = $1 AND f.partition_label = ANY($2) AND f.tsv @@ q.q
                     AND EXISTS (SELECT 1 FROM knowledge_fact_versions v
                                 WHERE v.corpus = f.corpus AND v.fact_id = f.id AND v.invalidated_at IS NULL)
                     AND ($5::text[] IS NULL OR EXISTS (SELECT 1 FROM knowledge_evidence ev WHERE ev.corpus = f.corpus AND ev.fact_id = f.id AND ev.source_id = ANY($5)))
                 ) hits
                 ORDER BY score DESC, id
                 LIMIT $4",
                &[&corpus, &partitions, &q, &(limit as i64), &sources],
            )
            .await
            .map_err(db)?;
        Ok(rows
            .iter()
            .map(|r| ScoredId {
                id: r.get(0),
                score: r.get(1),
            })
            .collect())
    }

    fn lexical_method(&self) -> String {
        "postgres-fts(english,ts_rank_cd)".into()
    }

    async fn projection(&self, corpus: &str, point_id: &str) -> Result<Option<Projection>> {
        let c = self.client().await?;
        let row = c
            .query_opt(
                "SELECT partition_label, text, ARRAY[source_id] FROM knowledge_chunks WHERE corpus = $1 AND id = $2
                 UNION ALL
                 SELECT partition_label, text, COALESCE((SELECT array_agg(DISTINCT ev.source_id ORDER BY ev.source_id) FROM knowledge_evidence ev WHERE ev.corpus = f.corpus AND ev.fact_id = f.id), '{}'::text[]) FROM knowledge_facts f WHERE corpus = $1 AND id = $2
                 LIMIT 1",
                &[&corpus, &point_id],
            )
            .await
            .map_err(db)?;
        Ok(row.map(|r| Projection {
            partition: r.get(0),
            text: r.get(1),
            sources: r.get(2),
        }))
    }

    async fn projections(&self, corpus: &str, ids: &[String]) -> Result<Vec<Option<Projection>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.client().await?;
        // Chunks win over facts for a shared id, as in `projection`.
        let mut found: BTreeMap<String, Projection> = BTreeMap::new();
        for r in c
            .query(
                "SELECT id, partition_label, text, 0 AS pick, ARRAY[source_id] FROM knowledge_chunks WHERE corpus = $1 AND id = ANY($2)
                 UNION ALL
                 SELECT id, partition_label, text, 1 AS pick, COALESCE((SELECT array_agg(DISTINCT ev.source_id ORDER BY ev.source_id) FROM knowledge_evidence ev WHERE ev.corpus = f.corpus AND ev.fact_id = f.id), '{}'::text[]) FROM knowledge_facts f WHERE corpus = $1 AND id = ANY($2)
                 ORDER BY pick DESC",
                &[&corpus, &ids],
            )
            .await
            .map_err(db)?
        {
            found.insert(
                r.get(0),
                Projection {
                    partition: r.get(1),
                    text: r.get(2),
                    sources: r.get(4),
                },
            );
        }
        Ok(ids.iter().map(|id| found.get(id).cloned()).collect())
    }

    async fn point_ids(&self, corpus: &str) -> Result<Vec<String>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT id FROM knowledge_chunks WHERE corpus = $1
             UNION ALL SELECT id FROM knowledge_facts WHERE corpus = $1
             ORDER BY 1",
            &[&corpus],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect())
    }

    async fn enqueue(&self, corpus: &str, jobs: &[NewJob], now: Time) -> Result<()> {
        let c = self.client().await?;
        enqueue_tx(&*c, corpus, jobs, now).await
    }

    async fn claim_jobs(&self, corpus: &str, now: Time, limit: usize) -> Result<Vec<Job>> {
        let c = self.client().await?;
        let lease_until = now + chrono::Duration::seconds(LEASE_SECONDS);
        let mut jobs: Vec<Job> = c
            .query(
                "UPDATE knowledge_jobs SET state = 'running', attempts = attempts + 1,
                        lease = nextval('knowledge_job_lease_seq'), next_attempt_at = $3
                 WHERE id IN (
                   SELECT id FROM knowledge_jobs
                   WHERE corpus = $1 AND state IN ('pending', 'retry_scheduled', 'running')
                     AND next_attempt_at <= $2
                   ORDER BY id LIMIT $4
                   FOR UPDATE SKIP LOCKED)
                 RETURNING *",
                &[&corpus, &now, &lease_until, &(limit as i64)],
            )
            .await
            .map_err(db)?
            .iter()
            .map(job_row)
            .collect();
        jobs.sort_by_key(|j| j.id);
        Ok(jobs)
    }

    async fn complete_job(&self, job: &Job, _now: Time) -> Result<bool> {
        let c = self.client().await?;
        let n = c
            .execute(
                "UPDATE knowledge_jobs SET state = 'succeeded', lease = NULL
                 WHERE id = $1 AND lease = $2 AND state = 'running'",
                &[&job.id, &job.lease],
            )
            .await
            .map_err(db)?;
        Ok(n == 1)
    }

    async fn complete_jobs(&self, jobs: &[Job], _now: Time) -> Result<Vec<bool>> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.client().await?;
        let ids: Vec<i64> = jobs.iter().map(|j| j.id).collect();
        let leases: Vec<Option<i64>> = jobs.iter().map(|j| j.lease).collect();
        let done: BTreeSet<i64> = c
            .query(
                "UPDATE knowledge_jobs AS k SET state = 'succeeded', lease = NULL
                 FROM unnest($1::int8[], $2::int8[]) AS j(id, lease)
                 WHERE k.id = j.id AND k.lease = j.lease AND k.state = 'running'
                 RETURNING k.id",
                &[&ids, &leases],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| r.get(0))
            .collect();
        Ok(ids.iter().map(|id| done.contains(id)).collect())
    }

    async fn fail_job(
        &self,
        job: &Job,
        error: &str,
        retryable: bool,
        now: Time,
    ) -> Result<JobState> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let Some(r) = tx
            .query_opt(
                "SELECT * FROM knowledge_jobs WHERE id = $1 FOR UPDATE",
                &[&job.id],
            )
            .await
            .map_err(db)?
        else {
            return Ok(JobState::Failed);
        };
        let cur = job_row(&r);
        if cur.state != JobState::Running || cur.lease != job.lease {
            return Ok(cur.state);
        }
        let (state, next) = if retryable && cur.attempts < cur.max_attempts {
            (JobState::RetryScheduled, now + retry_delay(cur.attempts))
        } else {
            (JobState::Failed, cur.next_attempt_at)
        };
        let msg: String = error.chars().take(500).collect();
        tx.execute(
            "UPDATE knowledge_jobs SET state = $2, next_attempt_at = $3, lease = NULL, last_error = $4
             WHERE id = $1",
            &[&job.id, &state.as_str(), &next, &msg],
        )
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(state)
    }

    async fn job_counts(
        &self,
        corpus: &str,
        partition: &str,
        ns: &str,
        succeeded: bool,
    ) -> Result<JobCounts> {
        let c = self.client().await?;
        let states: &[&str] = if succeeded {
            &[
                "pending",
                "running",
                "retry_scheduled",
                "succeeded",
                "failed",
            ]
        } else {
            &["pending", "running", "retry_scheduled", "failed"]
        };
        let mut counts = JobCounts::default();
        for r in c
            .query(
                "SELECT j.state, count(*) AS n FROM knowledge_jobs j
                 WHERE j.corpus = $1 AND j.state = ANY($4) AND CASE j.kind
                   WHEN 'extract' THEN EXISTS (SELECT 1 FROM knowledge_episodes e
                     WHERE e.corpus = j.corpus AND e.id = j.target AND e.partition_label = $2)
                   WHEN 'project' THEN EXISTS (SELECT 1 FROM knowledge_chunks k
                     WHERE k.corpus = j.corpus AND k.id = j.target AND k.partition_label = $2)
                     OR EXISTS (SELECT 1 FROM knowledge_facts f
                     WHERE f.corpus = j.corpus AND f.id = j.target AND f.partition_label = $2)
                   WHEN 'unproject' THEN split_part(j.target, chr(31), 1) = $3
                   ELSE false END
                 GROUP BY j.state",
                &[&corpus, &partition, &ns, &states],
            )
            .await
            .map_err(db)?
        {
            counts.add(JobState::parse(r.get("state")), r.get("n"));
        }
        Ok(counts)
    }

    async fn due_corpora(&self, now: Time, limit: usize) -> Result<Vec<String>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT DISTINCT corpus FROM knowledge_jobs
             WHERE state IN ('pending', 'retry_scheduled', 'running') AND next_attempt_at <= $1
             ORDER BY corpus LIMIT $2",
            &[&now, &(limit as i64)],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect())
    }

    async fn jobs(&self, corpus: &str) -> Result<Vec<Job>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT * FROM knowledge_jobs WHERE corpus = $1 ORDER BY id",
            &[&corpus],
        )
        .await
        .map_err(db)?
        .iter()
        .map(job_row)
        .collect())
    }

    async fn partitions(&self, corpus: &str) -> Result<Vec<(String, i64)>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT partition_label, generation FROM knowledge_partitions WHERE corpus = $1
             ORDER BY partition_label",
            &[&corpus],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| (r.get(0), r.get(1)))
        .collect())
    }

    async fn index_inputs(
        &self,
        corpus: &str,
        partition: &str,
        sources: Option<&[String]>,
    ) -> Result<IndexInputs> {
        let mut c = self.client().await?;
        let tx = c
            .build_transaction()
            .isolation_level(tokio_postgres::IsolationLevel::RepeatableRead)
            .read_only(true)
            .start()
            .await
            .map_err(db)?;
        let generation: i64 = tx
            .query_opt(
                "SELECT generation FROM knowledge_partitions WHERE corpus = $1 AND partition_label = $2",
                &[&corpus, &partition],
            )
            .await
            .map_err(db)?
            .map_or(0, |r| r.get(0));
        let filter = sources;
        let sources = tx
            .query(
                "SELECT s.id, s.title, s.collection,
                   (SELECT count(*) FROM knowledge_episodes e WHERE e.corpus = s.corpus AND e.source_id = s.id),
                   (SELECT count(DISTINCT v.fact_id) FROM knowledge_evidence v WHERE v.corpus = s.corpus AND v.source_id = s.id),
                   (SELECT min(e.occurred_at) FROM knowledge_episodes e WHERE e.corpus = s.corpus AND e.source_id = s.id),
                   (SELECT max(e.occurred_at) FROM knowledge_episodes e WHERE e.corpus = s.corpus AND e.source_id = s.id)
                 FROM knowledge_sources s WHERE s.corpus = $1 AND s.partition_label = $2
                   AND ($3::text[] IS NULL OR s.id = ANY($3))",
                &[&corpus, &partition, &filter],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| SourceStat {
                id: r.get(0),
                title: r.get(1),
                collection: r.get(2),
                episodes: r.get(3),
                facts: r.get(4),
                first_at: r.get(5),
                last_at: r.get(6),
            })
            .collect();
        let entities = tx
            .query(
                "SELECT en.id, en.name, en.kind,
                   COALESCE((SELECT array_agg(DISTINCT a.alias) FROM knowledge_entity_aliases a
                             WHERE a.corpus = en.corpus AND a.entity_id = en.id
                               AND ($3::text[] IS NULL OR a.source_id = ANY($3))), '{}'::text[]),
                   (SELECT count(*) FROM knowledge_facts f
                    WHERE f.corpus = en.corpus AND f.partition_label = en.partition_label
                      AND (f.subject = en.id OR f.object_entity = en.id)
                      AND ($3::text[] IS NULL OR EXISTS (SELECT 1 FROM knowledge_evidence ev WHERE ev.corpus = f.corpus AND ev.fact_id = f.id AND ev.source_id = ANY($3)))),
                   COALESCE((SELECT array_agg(DISTINCT v.source_id) FROM knowledge_evidence v
                             JOIN knowledge_facts f ON f.corpus = v.corpus AND f.id = v.fact_id
                             WHERE f.corpus = en.corpus AND f.partition_label = en.partition_label
                               AND (f.subject = en.id OR f.object_entity = en.id)
                               AND ($3::text[] IS NULL OR v.source_id = ANY($3))), '{}'::text[])
                 FROM knowledge_entities en WHERE en.corpus = $1 AND en.partition_label = $2",
                &[&corpus, &partition, &filter],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| EntityStat {
                id: r.get(0),
                name: r.get(1),
                kind: r.get(2),
                aliases: r.get(3),
                facts: r.get(4),
                sources: r.get(5),
            })
            .filter(|e: &EntityStat| filter.is_none() || e.facts > 0)
            .collect();
        tx.commit().await.map_err(db)?;
        Ok(IndexInputs {
            generation,
            sources,
            entities,
        })
    }

    async fn sources_in_collections(
        &self,
        corpus: &str,
        partitions: &[String],
        collections: &[String],
    ) -> Result<Vec<String>> {
        let c = self.client().await?;
        Ok(c.query(
            "SELECT id FROM knowledge_sources
             WHERE corpus = $1 AND partition_label = ANY($2) AND collection = ANY($3)
             ORDER BY id",
            &[&corpus, &partitions, &collections],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect())
    }

    async fn cached_index(&self, corpus: &str, partition: &str) -> Result<Option<(i64, String)>> {
        let c = self.client().await?;
        Ok(c.query_opt(
            "SELECT generation, body FROM knowledge_index_cache WHERE corpus = $1 AND partition_label = $2",
            &[&corpus, &partition],
        )
        .await
        .map_err(db)?
        .map(|r| (r.get(0), r.get(1))))
    }

    async fn store_index(
        &self,
        corpus: &str,
        partition: &str,
        generation: i64,
        body: &str,
    ) -> Result<bool> {
        let c = self.client().await?;
        // Only cache if the generation is still current, atomically.
        let n = c
            .execute(
                "INSERT INTO knowledge_index_cache (corpus, partition_label, generation, body)
                 SELECT $1, $2, $3, $4 FROM knowledge_partitions
                 WHERE corpus = $1 AND partition_label = $2 AND generation = $3
                 ON CONFLICT (corpus, partition_label)
                 DO UPDATE SET generation = EXCLUDED.generation, body = EXCLUDED.body",
                &[&corpus, &partition, &generation, &body],
            )
            .await
            .map_err(db)?;
        Ok(n == 1)
    }

    async fn purge_source(
        &self,
        corpus: &str,
        source_id: &str,
        now: Time,
    ) -> Result<Option<PurgeReport>> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let Some(src) = tx
            .query_opt(
                "SELECT * FROM knowledge_sources WHERE corpus = $1 AND id = $2 FOR UPDATE",
                &[&corpus, &source_id],
            )
            .await
            .map_err(db)?
            .map(|r| source_row(&r))
        else {
            return Ok(None);
        };
        let report = purge_tx(&tx, corpus, &src, now).await?;
        tx.commit().await.map_err(db)?;
        Ok(Some(report))
    }

    async fn erasure_journal(&self) -> Result<Vec<ErasureEntry>> {
        let c = self.client().await?;
        Ok(
            c.query("SELECT * FROM knowledge_erasure_journal ORDER BY seq", &[])
                .await
                .map_err(db)?
                .iter()
                .map(|r| ErasureEntry {
                    seq: r.get("seq"),
                    corpus_digest: r.get("corpus_digest"),
                    target_kind: r.get("target_kind"),
                    target_digest: r.get("target_digest"),
                    erased_at: r.get("erased_at"),
                    episodes: r.get("episodes"),
                    chunks: r.get("chunks"),
                    facts: r.get("facts"),
                })
                .collect(),
        )
    }

    async fn delete_corpus(&self, corpus: &str) -> Result<Vec<String>> {
        let mut c = self.client().await?;
        let tx = c.transaction().await.map_err(db)?;
        let parts: Vec<String> = tx
            .query(
                "DELETE FROM knowledge_partitions WHERE corpus = $1 RETURNING partition_label",
                &[&corpus],
            )
            .await
            .map_err(db)?
            .iter()
            .map(|r| r.get(0))
            .collect();
        tx.batch_execute("SET LOCAL lock_timeout = '30s'")
            .await
            .map_err(db)?;
        for table in [
            "knowledge_facts",
            "knowledge_entities",
            "knowledge_sources",
            "knowledge_jobs",
            "knowledge_index_cache",
        ] {
            tx.execute(
                &format!("DELETE FROM {table} WHERE corpus = $1"),
                &[&corpus],
            )
            .await
            .map_err(db)?;
        }
        tx.commit().await.map_err(db)?;
        Ok(parts)
    }
}

/// Delete a source and everything derived only from it, inside `tx`.
async fn purge_tx(
    tx: &Transaction<'_>,
    corpus: &str,
    src: &Source,
    now: Time,
) -> Result<PurgeReport> {
    let mut report = PurgeReport {
        partition: src.partition.clone(),
        ..Default::default()
    };
    let episode_ids: Vec<String> = tx
        .query(
            "SELECT id FROM knowledge_episodes WHERE corpus = $1 AND source_id = $2",
            &[&corpus, &src.id],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect();
    let chunk_ids: Vec<String> = tx
        .query(
            "SELECT id FROM knowledge_chunks WHERE corpus = $1 AND source_id = $2 ORDER BY id",
            &[&corpus, &src.id],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect();
    let touched: Vec<String> = tx
        .query(
            "SELECT DISTINCT fact_id FROM knowledge_evidence WHERE corpus = $1 AND source_id = $2",
            &[&corpus, &src.id],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect();
    // Cascades to episodes, chunks and their evidence.
    tx.execute(
        "DELETE FROM knowledge_sources WHERE corpus = $1 AND id = $2",
        &[&corpus, &src.id],
    )
    .await
    .map_err(db)?;
    tx.execute(
        "DELETE FROM knowledge_evidence WHERE corpus = $1 AND source_id = $2",
        &[&corpus, &src.id],
    )
    .await
    .map_err(db)?;
    let dead: BTreeSet<String> = tx
        .query(
            "DELETE FROM knowledge_facts f WHERE f.corpus = $1 AND f.id = ANY($2)
               AND NOT EXISTS (SELECT 1 FROM knowledge_evidence e
                               WHERE e.corpus = f.corpus AND e.fact_id = f.id)
             RETURNING f.id",
            &[&corpus, &touched],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect();
    report.episodes = episode_ids.len() as i64;
    report.chunks = chunk_ids.len() as i64;
    report.facts_deleted = dead.len() as i64;
    report.facts_kept = (touched.len() - dead.len()) as i64;
    report.kept_fact_ids = touched
        .iter()
        .filter(|id| !dead.contains(*id))
        .cloned()
        .collect();
    report.point_ids.extend(chunk_ids.iter().cloned());
    report.point_ids.extend(dead.iter().cloned());

    let dead_vec: Vec<String> = dead.iter().cloned().collect();
    let affected: Vec<String> = tx
        .query(
            "SELECT DISTINCT fact_id FROM knowledge_fact_versions WHERE corpus = $1 AND cause = ANY($2)",
            &[&corpus, &dead_vec],
        )
        .await
        .map_err(db)?
        .iter()
        .map(|r| r.get(0))
        .collect();
    for mut f in load_facts(tx, corpus, &affected).await? {
        splice_versions(&mut f, &dead);
        tx.execute(
            "DELETE FROM knowledge_fact_versions WHERE corpus = $1 AND fact_id = $2",
            &[&corpus, &f.id],
        )
        .await
        .map_err(db)?;
        for v in &f.versions {
            insert_version(tx, corpus, &f.id, v).await?;
        }
    }
    tx.execute(
        "DELETE FROM knowledge_entity_aliases WHERE corpus = $1 AND source_id = $2",
        &[&corpus, &src.id],
    )
    .await
    .map_err(db)?;
    tx.execute(
        "DELETE FROM knowledge_entities en WHERE en.corpus = $1 AND NOT EXISTS (
           SELECT 1 FROM knowledge_facts f
           WHERE f.corpus = en.corpus AND (f.subject = en.id OR f.object_entity = en.id))",
        &[&corpus],
    )
    .await
    .map_err(db)?;
    let targets: Vec<String> = episode_ids.iter().chain(&chunk_ids).cloned().collect();
    tx.execute(
        "DELETE FROM knowledge_jobs WHERE corpus = $1 AND target = ANY($2)
           AND state NOT IN ('succeeded', 'failed')",
        &[&corpus, &targets],
    )
    .await
    .map_err(db)?;
    bump(tx, corpus, &src.partition).await?;
    tx.execute(
        "DELETE FROM knowledge_index_cache WHERE corpus = $1 AND partition_label = $2",
        &[&corpus, &src.partition],
    )
    .await
    .map_err(db)?;
    tx.execute(
        "INSERT INTO knowledge_erasure_journal
         (corpus_digest, target_kind, target_digest, erased_at, episodes, chunks, facts)
         VALUES ($1, 'source', $2, $3, $4, $5, $6)",
        &[
            &digest_hex(&[corpus]),
            &digest_hex(&[corpus, &src.id]),
            &now,
            &report.episodes,
            &report.chunks,
            &report.facts_deleted,
        ],
    )
    .await
    .map_err(db)?;
    Ok(report)
}
