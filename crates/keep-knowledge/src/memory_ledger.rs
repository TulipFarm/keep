//! In-process ledger for tests, dry runs and small embedded uses. It follows
//! exactly the same contract as the Postgres ledger (the conformance suite
//! runs against both) but keeps nothing across restarts.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use async_trait::async_trait;
use keep_store::{Bm25Index, Bm25Params, LexicalIndex, ScoredId};

use crate::ledger::{
    ApplyOutcome, ChangeSet, EntityStat, GraphSlice, IndexInputs, IngestBatch, IngestReport,
    KnowledgeError, Ledger, MAX_JOB_ATTEMPTS, Projection, PurgeReport, Result, Revision,
    SourceStat, StoredVector, normalize_name, retry_delay, revision_plan,
};
use crate::model::{
    Chunk, Entity, Episode, ErasureEntry, Evidence, Fact, FactStatus, Job, JobCounts, JobKind,
    JobState, NewJob, Source, Time, digest_hex,
};

/// How long a claimed job stays leased before another worker may take it.
pub const LEASE_SECONDS: i64 = 300;

type Key = (String, String);

fn key(corpus: &str, id: &str) -> Key {
    (corpus.to_string(), id.to_string())
}

#[derive(Default)]
struct State {
    sources: BTreeMap<Key, Source>,
    episodes: BTreeMap<Key, Episode>,
    chunks: BTreeMap<Key, Chunk>,
    entities: BTreeMap<Key, Entity>,
    /// (corpus, entity id) -> alias -> sources that stated it.
    aliases: BTreeMap<Key, BTreeMap<String, BTreeSet<String>>>,
    facts: BTreeMap<Key, Fact>,
    jobs: Vec<Job>,
    next_job: i64,
    next_lease: i64,
    generations: BTreeMap<Key, i64>,
    index_cache: BTreeMap<Key, (i64, String)>,
    journal: Vec<ErasureEntry>,
    /// (corpus, chunk id) -> stored vector.
    vectors: BTreeMap<Key, Kept>,
}

struct Kept {
    source_id: String,
    digest: String,
    vector: Vec<f32>,
    removed_at: Option<Time>,
}

impl State {
    fn bump(&mut self, corpus: &str, partition: &str) {
        *self.generations.entry(key(corpus, partition)).or_insert(0) += 1;
    }

    fn enqueue(&mut self, corpus: &str, jobs: &[NewJob], now: Time) {
        for j in jobs {
            self.next_job += 1;
            self.jobs.push(Job {
                id: self.next_job,
                corpus: corpus.into(),
                kind: j.kind,
                target: j.target.clone(),
                state: JobState::Pending,
                attempts: 0,
                max_attempts: MAX_JOB_ATTEMPTS,
                next_attempt_at: now,
                lease: None,
                last_error: None,
            });
        }
    }

    /// Delete a source and everything derived only from it.
    fn purge(&mut self, corpus: &str, source_id: &str, now: Time) -> Option<PurgeReport> {
        let src = self.sources.remove(&key(corpus, source_id))?;
        Some(self.purge_episodes(corpus, &src, None, true, now))
    }

    /// Delete episodes of `src` (all of them with `None`) and everything
    /// derived only from them. With `forget` the source's stored vectors go
    /// at once; otherwise removed chunks' vectors wait out the grace period.
    fn purge_episodes(
        &mut self,
        corpus: &str,
        src: &Source,
        only: Option<&BTreeSet<String>>,
        forget: bool,
        now: Time,
    ) -> PurgeReport {
        let s = self;
        let source_id = src.id.as_str();
        let mut report = PurgeReport {
            partition: src.partition.clone(),
            ..Default::default()
        };
        let episode_ids: BTreeSet<String> = s
            .episodes
            .values()
            .filter(|e| e.corpus == corpus && e.source_id == source_id)
            .filter(|e| only.is_none_or(|o| o.contains(&e.id)))
            .map(|e| e.id.clone())
            .collect();
        for id in &episode_ids {
            s.episodes.remove(&key(corpus, id));
        }
        let survivors = s
            .episodes
            .values()
            .any(|e| e.corpus == corpus && e.source_id == source_id);
        report.episodes = episode_ids.len() as i64;
        let chunk_ids: Vec<String> = s
            .chunks
            .values()
            .filter(|c| c.corpus == corpus && episode_ids.contains(&c.episode_id))
            .map(|c| c.id.clone())
            .collect();
        for id in &chunk_ids {
            s.chunks.remove(&key(corpus, id));
        }
        report.chunks = chunk_ids.len() as i64;
        report.point_ids.extend(chunk_ids.iter().cloned());
        if forget {
            s.vectors
                .retain(|(c, _), v| c != corpus || v.source_id != source_id);
        } else {
            for id in &chunk_ids {
                if let Some(v) = s.vectors.get_mut(&key(corpus, id)) {
                    v.removed_at = Some(now);
                }
            }
        }

        let mut dead = BTreeSet::new();
        for f in s.facts.values_mut().filter(|f| f.corpus == corpus) {
            let before = f.evidence.len();
            f.evidence
                .retain(|e| e.source_id != source_id || !episode_ids.contains(&e.episode_id));
            if f.evidence.len() != before {
                if f.evidence.is_empty() {
                    dead.insert(f.id.clone());
                } else {
                    report.facts_kept += 1;
                    report.kept_fact_ids.push(f.id.clone());
                }
            }
        }
        for id in &dead {
            s.facts.remove(&key(corpus, id));
        }
        report.facts_deleted = dead.len() as i64;
        report.point_ids.extend(dead.iter().cloned());
        for f in s.facts.values_mut().filter(|f| f.corpus == corpus) {
            splice_versions(f, &dead);
        }

        // Aliases are recorded per source, not per episode: they stay while
        // any episode of the source does.
        if !survivors {
            for (_, m) in s.aliases.iter_mut().filter(|((c, _), _)| c == corpus) {
                for srcs in m.values_mut() {
                    srcs.remove(source_id);
                }
                m.retain(|_, srcs| !srcs.is_empty());
            }
        }
        let referenced: BTreeSet<String> = s
            .facts
            .values()
            .filter(|f| f.corpus == corpus)
            .flat_map(|f| std::iter::once(f.subject.clone()).chain(f.object_entity.clone()))
            .collect();
        let orphans: Vec<Key> = s
            .entities
            .iter()
            .filter(|((c, id), _)| c == corpus && !referenced.contains(id))
            .map(|(k, _)| k.clone())
            .collect();
        for k in orphans {
            s.entities.remove(&k);
            s.aliases.remove(&k);
        }
        let chunk_set: BTreeSet<&String> = chunk_ids.iter().collect();
        s.jobs.retain(|j| {
            j.corpus != corpus
                || matches!(j.state, JobState::Succeeded | JobState::Failed)
                || !(episode_ids.contains(&j.target) || chunk_set.contains(&j.target))
        });
        s.bump(corpus, &src.partition);
        s.index_cache.remove(&key(corpus, &src.partition));
        let seq = s.journal.len() as i64 + 1;
        s.journal.push(ErasureEntry {
            seq,
            corpus_digest: digest_hex(&[corpus]),
            target_kind: "source".into(),
            target_digest: digest_hex(&[corpus, source_id]),
            erased_at: now,
            episodes: report.episodes,
            chunks: report.chunks,
            facts: report.facts_deleted,
        });
        report
    }

    fn entity_with_aliases(&self, e: &Entity) -> Entity {
        self.entity_with_aliases_from(e, None)
    }

    /// Aliases of `e`; with `sources`, only those stated by one of them.
    fn aliases_of(&self, e: &Entity, sources: Option<&[String]>) -> BTreeSet<String> {
        self.aliases
            .get(&key(&e.corpus, &e.id))
            .map(|m| {
                m.iter()
                    .filter(|(_, by)| sources.is_none_or(|s| by.iter().any(|x| s.contains(x))))
                    .map(|(a, _)| a.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn entity_with_aliases_from(&self, e: &Entity, sources: Option<&[String]>) -> Entity {
        let mut e = e.clone();
        e.aliases = self.aliases_of(&e, sources);
        e
    }

    fn entity_matches(&self, e: &Entity, normalized: &str, sources: Option<&[String]>) -> bool {
        normalize_name(&e.name) == normalized
            || self
                .aliases_of(e, sources)
                .iter()
                .any(|a| normalize_name(a) == normalized)
    }
}

#[derive(Default)]
pub struct MemoryLedger {
    state: Mutex<State>,
}

impl MemoryLedger {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("ledger lock poisoned")
    }
}

/// Whether a fact has evidence from one of `sources` (any fact without a filter).
fn supported_by(f: &Fact, sources: Option<&[String]>) -> bool {
    sources.is_none_or(|ids| f.evidence.iter().any(|e| ids.contains(&e.source_id)))
}

fn last_recorded(f: &Fact) -> Time {
    f.versions
        .iter()
        .map(|v| v.recorded_at)
        .chain(f.evidence.iter().map(|e| e.recorded_at))
        .max()
        .unwrap_or_default()
}

#[async_trait]
impl Ledger for MemoryLedger {
    fn backend(&self) -> &'static str {
        "memory"
    }

    async fn ingest(&self, batch: IngestBatch) -> Result<IngestReport> {
        let mut s = self.lock();
        let src = &batch.source;
        let corpus = src.corpus.as_str();
        let mut report = IngestReport::default();
        let plan = match s.sources.get(&key(corpus, &src.id)) {
            Some(existing) => {
                let held: BTreeSet<(String, String)> = s
                    .episodes
                    .values()
                    .filter(|e| e.corpus == corpus && e.source_id == src.id)
                    .map(|e| (e.id.clone(), e.content_hash.clone()))
                    .collect();
                revision_plan(existing, &batch, held)?
            }
            None => Revision::default(),
        };
        let purging = |id: &str| plan.full || plan.removed.contains(id);
        let mut fresh = BTreeSet::new();
        for e in &batch.episodes {
            match s.episodes.get(&key(corpus, &e.id)) {
                // Purged below before the new revision is inserted.
                Some(old) if old.source_id == src.id && purging(&e.id) => {
                    fresh.insert(e.id.clone());
                }
                Some(old) if old.content_hash == e.content_hash && old.source_id == e.source_id => {
                    report.episodes_unchanged += 1;
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
        if plan.full {
            report.purged = s
                .sources
                .remove(&key(corpus, &src.id))
                .map(|old| s.purge_episodes(corpus, &old, None, false, batch.now));
            report.episodes_removed = report.purged.as_ref().map_or(0, |p| p.episodes as usize);
        } else if !plan.removed.is_empty() {
            let purged = s.purge_episodes(corpus, src, Some(&plan.removed), false, batch.now);
            report.episodes_removed = purged.episodes as usize;
            report.purged = Some(purged);
        }
        report.replaced = plan.full || !plan.removed.is_empty() || plan.metadata;
        match s.sources.get_mut(&key(corpus, &src.id)) {
            Some(held) if plan.metadata => {
                *held = Source {
                    created_at: held.created_at,
                    ..src.clone()
                }
            }
            Some(_) => {}
            None => {
                s.sources.insert(key(corpus, &src.id), src.clone());
                report.source_created = !report.replaced;
            }
        }
        let mut jobs = Vec::new();
        for e in batch.episodes.iter().filter(|e| fresh.contains(&e.id)) {
            s.episodes.insert(key(corpus, &e.id), e.clone());
            jobs.push(NewJob {
                kind: JobKind::Extract,
                target: e.id.clone(),
            });
            report.episodes_added += 1;
        }
        for c in batch
            .chunks
            .iter()
            .filter(|c| fresh.contains(&c.episode_id))
        {
            s.chunks.insert(key(corpus, &c.id), c.clone());
            if let Some(v) = s.vectors.get_mut(&key(corpus, &c.id)) {
                v.removed_at = None;
            }
            jobs.push(NewJob {
                kind: JobKind::Project,
                target: c.id.clone(),
            });
            report.chunks_added += 1;
        }
        report.jobs_enqueued = jobs.len();
        s.enqueue(corpus, &jobs, batch.now);
        if report.source_created || report.replaced || report.episodes_added > 0 {
            s.bump(corpus, &src.partition);
        }
        Ok(report)
    }

    async fn source(&self, corpus: &str, id: &str) -> Result<Option<Source>> {
        Ok(self.lock().sources.get(&key(corpus, id)).cloned())
    }

    async fn episode(&self, corpus: &str, id: &str) -> Result<Option<Episode>> {
        Ok(self.lock().episodes.get(&key(corpus, id)).cloned())
    }

    async fn source_episodes(&self, corpus: &str, source_id: &str) -> Result<Vec<Episode>> {
        let mut eps: Vec<Episode> = self
            .lock()
            .episodes
            .values()
            .filter(|e| e.corpus == corpus && e.source_id == source_id)
            .cloned()
            .collect();
        eps.sort_by_key(|e| e.ordinal);
        Ok(eps)
    }

    async fn chunks(&self, corpus: &str, ids: &[String]) -> Result<Vec<Chunk>> {
        let s = self.lock();
        Ok(ids
            .iter()
            .filter_map(|id| s.chunks.get(&key(corpus, id)).cloned())
            .collect())
    }

    async fn facts(&self, corpus: &str, ids: &[String]) -> Result<Vec<Fact>> {
        let s = self.lock();
        Ok(ids
            .iter()
            .filter_map(|id| s.facts.get(&key(corpus, id)).cloned())
            .collect())
    }

    async fn evidence(&self, corpus: &str, id: &str) -> Result<Option<Evidence>> {
        let s = self.lock();
        Ok(s.facts
            .values()
            .filter(|f| f.corpus == corpus)
            .flat_map(|f| f.evidence.iter())
            .find(|e| e.id == id)
            .cloned())
    }

    async fn entity(&self, corpus: &str, id: &str) -> Result<Option<Entity>> {
        let s = self.lock();
        Ok(s.entities
            .get(&key(corpus, id))
            .map(|e| s.entity_with_aliases(e)))
    }

    async fn find_entity(
        &self,
        corpus: &str,
        partition: &str,
        normalized: &str,
    ) -> Result<Option<Entity>> {
        let s = self.lock();
        Ok(s.entities
            .values()
            .filter(|e| e.corpus == corpus && e.partition == partition)
            .find(|e| s.entity_matches(e, normalized, None))
            .map(|e| s.entity_with_aliases(e)))
    }

    async fn entities_named(
        &self,
        corpus: &str,
        partitions: &[String],
        normalized: &str,
        sources: Option<&[String]>,
    ) -> Result<Vec<Entity>> {
        let s = self.lock();
        Ok(s.entities
            .values()
            .filter(|e| e.corpus == corpus && partitions.contains(&e.partition))
            .filter(|e| s.entity_matches(e, normalized, sources))
            .map(|e| s.entity_with_aliases_from(e, sources))
            .collect())
    }

    async fn aliases_from(
        &self,
        corpus: &str,
        entity_id: &str,
        sources: &[String],
    ) -> Result<BTreeSet<String>> {
        let s = self.lock();
        Ok(s.entities
            .get(&key(corpus, entity_id))
            .map(|e| s.aliases_of(e, Some(sources)))
            .unwrap_or_default())
    }

    async fn current_facts(
        &self,
        corpus: &str,
        partition: &str,
        subject: &str,
        predicate: &str,
    ) -> Result<Vec<Fact>> {
        let s = self.lock();
        Ok(s.facts
            .values()
            .filter(|f| {
                f.corpus == corpus
                    && f.partition == partition
                    && f.subject == subject
                    && f.predicate == predicate
                    && f.current().is_some()
            })
            .cloned()
            .collect())
    }

    async fn facts_about(
        &self,
        corpus: &str,
        partitions: &[String],
        entity_id: &str,
        limit: usize,
        sources: Option<&[String]>,
    ) -> Result<Vec<Fact>> {
        let s = self.lock();
        let mut out: Vec<Fact> = s
            .facts
            .values()
            .filter(|f| f.corpus == corpus && partitions.contains(&f.partition))
            .filter(|f| f.subject == entity_id || f.object_entity.as_deref() == Some(entity_id))
            .filter(|f| supported_by(f, sources))
            .cloned()
            .collect();
        out.sort_by(|a, b| {
            last_recorded(b)
                .cmp(&last_recorded(a))
                .then_with(|| a.id.cmp(&b.id))
        });
        out.truncate(limit);
        Ok(out)
    }

    async fn apply(&self, c: ChangeSet) -> Result<ApplyOutcome> {
        let mut s = self.lock();
        let corpus = c.corpus.as_str();
        let current = |id: &str, hash: &str| {
            s.episodes
                .get(&key(corpus, id))
                .is_some_and(|e| e.content_hash == hash)
        };
        if !s.episodes.contains_key(&key(corpus, &c.episode_id))
            || !c.episodes.iter().all(|(id, hash)| current(id, hash))
        {
            return Ok(ApplyOutcome::EpisodeGone);
        }
        if c.new_facts
            .iter()
            .any(|f| s.facts.contains_key(&key(corpus, &f.id)))
        {
            return Ok(ApplyOutcome::Conflict);
        }
        for v in &c.versions {
            match s
                .facts
                .get(&key(corpus, &v.fact_id))
                .and_then(|f| f.current())
            {
                Some(cur) if cur.version == v.expected_current => {}
                _ => return Ok(ApplyOutcome::Conflict),
            }
        }
        let new_ids: BTreeSet<&str> = c.new_facts.iter().map(|f| f.id.as_str()).collect();
        if c.evidence.iter().any(|e| {
            !new_ids.contains(e.fact_id.as_str()) && !s.facts.contains_key(&key(corpus, &e.fact_id))
        }) {
            return Ok(ApplyOutcome::Conflict);
        }
        for e in &c.entities {
            s.entities
                .entry(key(corpus, &e.id))
                .or_insert_with(|| Entity {
                    corpus: corpus.into(),
                    id: e.id.clone(),
                    partition: c.partition.clone(),
                    name: e.name.clone(),
                    kind: e.kind.clone(),
                    aliases: Default::default(),
                });
        }
        for a in &c.aliases {
            s.aliases
                .entry(key(corpus, &a.entity_id))
                .or_default()
                .entry(a.alias.clone())
                .or_default()
                .insert(a.source_id.clone());
        }
        for f in &c.new_facts {
            s.facts.insert(key(corpus, &f.id), f.clone());
        }
        for e in &c.evidence {
            let f = s.facts.get_mut(&key(corpus, &e.fact_id)).expect("checked");
            if !f.evidence.iter().any(|x| x.id == e.id) {
                f.evidence.push(e.clone());
                // Same order as the Postgres ledger.
                f.evidence
                    .sort_by(|a, b| (a.recorded_at, &a.id).cmp(&(b.recorded_at, &b.id)));
            }
        }
        for v in &c.versions {
            let f = s.facts.get_mut(&key(corpus, &v.fact_id)).expect("checked");
            if let Some(cur) = f.versions.iter_mut().rev().find(|x| x.is_current()) {
                cur.invalidated_at = Some(v.version.recorded_at);
            }
            f.versions.push(v.version.clone());
        }
        s.enqueue(corpus, &c.jobs, c.now);
        if !c.is_empty() {
            s.bump(corpus, &c.partition);
        }
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
        let docs: Vec<(String, String)> = {
            let s = self.lock();
            let chunks = s
                .chunks
                .values()
                .filter(|c| c.corpus == corpus && partitions.contains(&c.partition))
                .filter(|c| sources.is_none_or(|ids| ids.contains(&c.source_id)))
                .map(|c| (c.id.clone(), c.text.clone()));
            let facts = s
                .facts
                .values()
                .filter(|f| {
                    f.corpus == corpus
                        && partitions.contains(&f.partition)
                        && f.current().is_some()
                        && supported_by(f, sources)
                })
                .map(|f| (f.id.clone(), f.text.clone()));
            chunks.chain(facts).collect()
        };
        Ok(Bm25Index::build(Bm25Params::default(), docs).search(query, limit))
    }

    fn lexical_method(&self) -> String {
        Bm25Index::build(Bm25Params::default(), Vec::new()).method()
    }

    async fn projection(&self, corpus: &str, point_id: &str) -> Result<Option<Projection>> {
        let s = self.lock();
        if let Some(c) = s.chunks.get(&key(corpus, point_id)) {
            return Ok(Some(Projection {
                partition: c.partition.clone(),
                text: c.text.clone(),
                sources: vec![c.source_id.clone()],
            }));
        }
        Ok(s.facts.get(&key(corpus, point_id)).map(|f| Projection {
            partition: f.partition.clone(),
            text: f.text.clone(),
            sources: f
                .evidence
                .iter()
                .map(|e| e.source_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        }))
    }

    async fn point_ids(&self, corpus: &str) -> Result<Vec<String>> {
        let s = self.lock();
        Ok(s.chunks
            .keys()
            .chain(s.facts.keys())
            .filter(|k| k.0 == corpus)
            .map(|k| k.1.clone())
            .collect())
    }

    async fn enqueue(&self, corpus: &str, jobs: &[NewJob], now: Time) -> Result<()> {
        self.lock().enqueue(corpus, jobs, now);
        Ok(())
    }

    async fn claim_jobs(&self, corpus: &str, now: Time, limit: usize) -> Result<Vec<Job>> {
        let mut s = self.lock();
        let mut out = Vec::new();
        let mut lease = s.next_lease;
        for j in s.jobs.iter_mut() {
            if out.len() >= limit {
                break;
            }
            let due = matches!(
                j.state,
                JobState::Pending | JobState::RetryScheduled | JobState::Running
            ) && j.next_attempt_at <= now;
            if j.corpus == corpus && due {
                lease += 1;
                j.state = JobState::Running;
                j.attempts += 1;
                j.lease = Some(lease);
                j.next_attempt_at = now + chrono::Duration::seconds(LEASE_SECONDS);
                out.push(j.clone());
            }
        }
        s.next_lease = lease;
        Ok(out)
    }

    async fn claim_source_jobs(
        &self,
        corpus: &str,
        source_id: &str,
        now: Time,
        limit: usize,
    ) -> Result<Vec<Job>> {
        let mut guard = self.lock();
        let s = &mut *guard;
        let mut mine: BTreeSet<&String> = s
            .episodes
            .values()
            .filter(|e| e.corpus == corpus && e.source_id == source_id)
            .map(|e| &e.id)
            .collect();
        mine.extend(
            s.chunks
                .values()
                .filter(|c| c.corpus == corpus && c.source_id == source_id)
                .map(|c| &c.id),
        );
        mine.extend(
            s.facts
                .values()
                .filter(|f| {
                    f.corpus == corpus && f.evidence.iter().any(|e| e.source_id == source_id)
                })
                .map(|f| &f.id),
        );
        let mut out = Vec::new();
        let mut lease = s.next_lease;
        for j in s.jobs.iter_mut() {
            if out.len() >= limit {
                break;
            }
            let due = matches!(
                j.state,
                JobState::Pending | JobState::RetryScheduled | JobState::Running
            ) && j.next_attempt_at <= now;
            if j.corpus == corpus && due && mine.contains(&j.target) {
                lease += 1;
                j.state = JobState::Running;
                j.attempts += 1;
                j.lease = Some(lease);
                j.next_attempt_at = now + chrono::Duration::seconds(LEASE_SECONDS);
                out.push(j.clone());
            }
        }
        s.next_lease = lease;
        Ok(out)
    }

    async fn release_jobs(&self, jobs: &[Job], now: Time) -> Result<()> {
        let mut s = self.lock();
        for job in jobs {
            if let Some(j) = s.jobs.iter_mut().find(|j| j.id == job.id)
                && j.state == JobState::Running
                && j.lease == job.lease
            {
                j.state = JobState::Pending;
                j.attempts = (j.attempts - 1).max(0);
                j.lease = None;
                j.next_attempt_at = now;
            }
        }
        Ok(())
    }

    async fn complete_job(&self, job: &Job, _now: Time) -> Result<bool> {
        let mut s = self.lock();
        match s.jobs.iter_mut().find(|j| j.id == job.id) {
            Some(j) if j.state == JobState::Running && j.lease == job.lease => {
                j.state = JobState::Succeeded;
                j.lease = None;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    async fn fail_job(
        &self,
        job: &Job,
        error: &str,
        retryable: bool,
        now: Time,
    ) -> Result<JobState> {
        let mut s = self.lock();
        let Some(j) = s.jobs.iter_mut().find(|j| j.id == job.id) else {
            return Ok(JobState::Failed);
        };
        if j.state != JobState::Running || j.lease != job.lease {
            return Ok(j.state);
        }
        j.lease = None;
        j.last_error = Some(error.chars().take(500).collect());
        if retryable && j.attempts < j.max_attempts {
            j.state = JobState::RetryScheduled;
            j.next_attempt_at = now + retry_delay(j.attempts);
        } else {
            j.state = JobState::Failed;
        }
        Ok(j.state)
    }

    async fn job_counts(
        &self,
        corpus: &str,
        partition: &str,
        ns: &str,
        succeeded: bool,
    ) -> Result<JobCounts> {
        let s = self.lock();
        let mut counts = JobCounts::default();
        for j in s
            .jobs
            .iter()
            .filter(|j| j.corpus == corpus && (succeeded || j.state != JobState::Succeeded))
        {
            let k = key(corpus, &j.target);
            let mine = match j.kind {
                JobKind::Extract => s.episodes.get(&k).is_some_and(|e| e.partition == partition),
                JobKind::Project => {
                    s.chunks.get(&k).is_some_and(|c| c.partition == partition)
                        || s.facts.get(&k).is_some_and(|f| f.partition == partition)
                }
                JobKind::Unproject => j.target.split('\u{1f}').next() == Some(ns),
            };
            if mine {
                counts.add(j.state, 1);
            }
        }
        Ok(counts)
    }

    async fn due_corpora(&self, now: Time, limit: usize) -> Result<Vec<String>> {
        let mut out = std::collections::BTreeSet::new();
        for j in &self.lock().jobs {
            if matches!(
                j.state,
                JobState::Pending | JobState::RetryScheduled | JobState::Running
            ) && j.next_attempt_at <= now
            {
                out.insert(j.corpus.clone());
            }
        }
        Ok(out.into_iter().take(limit).collect())
    }

    async fn jobs(&self, corpus: &str) -> Result<Vec<Job>> {
        Ok(self
            .lock()
            .jobs
            .iter()
            .filter(|j| j.corpus == corpus)
            .cloned()
            .collect())
    }

    async fn partitions(&self, corpus: &str) -> Result<Vec<(String, i64)>> {
        Ok(self
            .lock()
            .generations
            .iter()
            .filter(|((c, _), _)| c == corpus)
            .map(|((_, p), g)| (p.clone(), *g))
            .collect())
    }

    async fn index_inputs(
        &self,
        corpus: &str,
        partition: &str,
        filter: Option<&[String]>,
    ) -> Result<IndexInputs> {
        let s = self.lock();
        let allowed = |id: &String| filter.is_none_or(|ids| ids.contains(id));
        let generation = s
            .generations
            .get(&key(corpus, partition))
            .copied()
            .unwrap_or(0);
        let facts: Vec<&Fact> = s
            .facts
            .values()
            .filter(|f| f.corpus == corpus && f.partition == partition)
            .filter(|f| supported_by(f, filter))
            .collect();
        let sources = s
            .sources
            .values()
            .filter(|x| x.corpus == corpus && x.partition == partition && allowed(&x.id))
            .map(|src| {
                let eps: Vec<&Episode> = s
                    .episodes
                    .values()
                    .filter(|e| e.corpus == corpus && e.source_id == src.id)
                    .collect();
                SourceStat {
                    id: src.id.clone(),
                    title: src.title.clone(),
                    collection: src.collection.clone(),
                    episodes: eps.len() as i64,
                    facts: facts
                        .iter()
                        .filter(|f| f.evidence.iter().any(|e| e.source_id == src.id))
                        .count() as i64,
                    first_at: eps.iter().filter_map(|e| e.occurred_at).min(),
                    last_at: eps.iter().filter_map(|e| e.occurred_at).max(),
                }
            })
            .collect();
        let entities = s
            .entities
            .values()
            .filter(|e| e.corpus == corpus && e.partition == partition)
            .map(|e| {
                let about: Vec<&&Fact> = facts
                    .iter()
                    .filter(|f| f.subject == e.id || f.object_entity.as_deref() == Some(&e.id))
                    .collect();
                let srcs: BTreeSet<String> = about
                    .iter()
                    .flat_map(|f| f.evidence.iter().map(|x| x.source_id.clone()))
                    .filter(|id| allowed(id))
                    .collect();
                EntityStat {
                    id: e.id.clone(),
                    name: e.name.clone(),
                    kind: e.kind.clone(),
                    aliases: s.aliases_of(e, filter).into_iter().collect(),
                    facts: about.len() as i64,
                    sources: srcs.into_iter().collect(),
                }
            })
            .filter(|e| filter.is_none() || e.facts > 0)
            .collect();
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
        Ok(self
            .lock()
            .sources
            .values()
            .filter(|x| {
                x.corpus == corpus
                    && partitions.contains(&x.partition)
                    && collections.contains(&x.collection)
            })
            .map(|x| x.id.clone())
            .collect())
    }

    async fn graph(&self, corpus: &str, partitions: &[String], limit: usize) -> Result<GraphSlice> {
        let s = self.lock();
        let readable = |corpus_of: &str, partition: &str| {
            corpus_of == corpus && partitions.iter().any(|p| p == partition)
        };
        let sources = s
            .sources
            .values()
            .filter(|x| readable(&x.corpus, &x.partition))
            .cloned()
            .collect();
        let mut facts: Vec<Fact> = s
            .facts
            .values()
            .filter(|f| readable(&f.corpus, &f.partition) && f.current().is_some())
            .cloned()
            .collect();
        facts.sort_by(|a, b| {
            last_recorded(b)
                .cmp(&last_recorded(a))
                .then_with(|| a.id.cmp(&b.id))
        });
        let truncated = facts.len() > limit;
        facts.truncate(limit);
        let named: BTreeSet<&str> = facts
            .iter()
            .flat_map(|f| std::iter::once(f.subject.as_str()).chain(f.object_entity.as_deref()))
            .collect();
        let entities = named
            .into_iter()
            .filter_map(|id| s.entities.get(&key(corpus, id)))
            .map(|e| s.entity_with_aliases_from(e, None))
            .collect();
        Ok(GraphSlice {
            sources,
            facts,
            entities,
            truncated,
        })
    }

    async fn cached_index(&self, corpus: &str, partition: &str) -> Result<Option<(i64, String)>> {
        Ok(self
            .lock()
            .index_cache
            .get(&key(corpus, partition))
            .cloned())
    }

    async fn store_index(
        &self,
        corpus: &str,
        partition: &str,
        generation: i64,
        body: &str,
    ) -> Result<bool> {
        let mut s = self.lock();
        let current = s.generations.get(&key(corpus, partition)).copied();
        if current != Some(generation) {
            return Ok(false);
        }
        s.index_cache
            .insert(key(corpus, partition), (generation, body.to_string()));
        Ok(true)
    }

    async fn stored_vectors(
        &self,
        corpus: &str,
        keys: &[(String, String)],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let s = self.lock();
        Ok(keys
            .iter()
            .map(|(id, digest)| {
                s.vectors
                    .get(&key(corpus, id))
                    .filter(|v| v.digest == *digest)
                    .map(|v| v.vector.clone())
            })
            .collect())
    }

    async fn store_vectors(
        &self,
        corpus: &str,
        vectors: &[StoredVector],
        _now: Time,
    ) -> Result<()> {
        let mut s = self.lock();
        for v in vectors {
            let Some(source_id) = s
                .chunks
                .get(&key(corpus, &v.point_id))
                .map(|c| c.source_id.clone())
            else {
                continue;
            };
            s.vectors.insert(
                key(corpus, &v.point_id),
                Kept {
                    source_id,
                    digest: v.digest.clone(),
                    vector: v.vector.clone(),
                    removed_at: None,
                },
            );
        }
        Ok(())
    }

    async fn sweep_vectors(&self, removed_before: Time) -> Result<usize> {
        let mut s = self.lock();
        let before = s.vectors.len();
        s.vectors
            .retain(|_, v| v.removed_at.is_none_or(|t| t >= removed_before));
        Ok(before - s.vectors.len())
    }

    async fn purge_source(
        &self,
        corpus: &str,
        source_id: &str,
        now: Time,
    ) -> Result<Option<PurgeReport>> {
        Ok(self.lock().purge(corpus, source_id, now))
    }

    async fn erasure_journal(&self) -> Result<Vec<ErasureEntry>> {
        Ok(self.lock().journal.clone())
    }

    async fn delete_corpus(&self, corpus: &str) -> Result<Vec<String>> {
        let mut s = self.lock();
        let keep = |k: &Key| k.0 != corpus;
        s.sources.retain(|k, _| keep(k));
        s.episodes.retain(|k, _| keep(k));
        s.chunks.retain(|k, _| keep(k));
        s.entities.retain(|k, _| keep(k));
        s.aliases.retain(|k, _| keep(k));
        s.facts.retain(|k, _| keep(k));
        s.index_cache.retain(|k, _| keep(k));
        s.vectors.retain(|k, _| keep(k));
        s.jobs.retain(|j| j.corpus != corpus);
        let parts: Vec<String> = s
            .generations
            .keys()
            .filter(|k| k.0 == corpus)
            .map(|k| k.1.clone())
            .collect();
        s.generations.retain(|k, _| keep(k));
        Ok(parts)
    }
}

/// Remove the effects of deleted facts on surviving version chains. A
/// version caused by a deleted fact is dropped and the version before it
/// becomes current again; if it was the first version, it is reopened.
pub fn splice_versions(f: &mut Fact, dead: &BTreeSet<String>) {
    let mut i = 0;
    while i < f.versions.len() {
        let caused_by_dead = f.versions[i]
            .cause
            .as_ref()
            .is_some_and(|c| dead.contains(c));
        if !caused_by_dead {
            i += 1;
            continue;
        }
        if i == 0 {
            let v = &mut f.versions[0];
            v.valid_to = None;
            v.status = FactStatus::Active;
            v.cause = None;
            i += 1;
        } else {
            let removed = f.versions.remove(i);
            f.versions[i - 1].invalidated_at = removed.invalidated_at;
        }
    }
}
