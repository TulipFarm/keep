//! In-process snapshots of a corpus for visibility-first search.
//!
//! [`crate::engine::LexicalMode::Bm25`] ranks only items the reader may see
//! at the requested time. It loads every chunk and fact of a corpus once per
//! partition generation, keeps a few corpora in memory, and builds a BM25
//! index per distinct visible set, so a question asked before later sessions
//! were recorded never loses candidates to items it cannot see.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use keep_store::{Bm25Index, Bm25Params};

use crate::model::{Access, Chunk, Fact};
use crate::temporal::{AsOf, chunk_visible, fact_as_of};

/// Corpora kept loaded at once.
const SNAPSHOTS: usize = 8;
/// BM25 indexes kept per corpus, one per distinct visible set.
const INDEXES: usize = 16;

pub(crate) struct Snapshot {
    /// Partitions and generations the snapshot was loaded at.
    pub key: Vec<(String, i64)>,
    pub chunks: Vec<Chunk>,
    pub facts: Vec<Fact>,
    indexes: Mutex<VecDeque<(Vec<u32>, Arc<Bm25Index>)>>,
}

/// What one reader may see in a snapshot: indexes into `chunks` (first) and
/// `facts` (offset by the chunk count), in snapshot order.
pub(crate) struct Visible {
    pub items: Vec<u32>,
    /// Point id to position in `items`.
    pub by_id: HashMap<String, u32>,
    /// Items in readable partitions that this reader cannot see yet.
    pub hidden: usize,
}

impl Snapshot {
    pub fn new(key: Vec<(String, i64)>, mut chunks: Vec<Chunk>, mut facts: Vec<Fact>) -> Self {
        chunks.sort_by(|a, b| a.id.cmp(&b.id));
        facts.sort_by(|a, b| a.id.cmp(&b.id));
        Self {
            key,
            chunks,
            facts,
            indexes: Mutex::new(VecDeque::new()),
        }
    }

    pub fn id(&self, item: u32) -> &str {
        let i = item as usize;
        match self.chunks.get(i) {
            Some(c) => &c.id,
            None => &self.facts[i - self.chunks.len()].id,
        }
    }

    /// With `sources`, only chunks of those sources and facts with evidence
    /// from them (judged on that evidence alone) are visible.
    pub fn visible(
        &self,
        parts: &[String],
        access: &Access,
        sources: Option<&BTreeSet<String>>,
        as_of: AsOf,
    ) -> Visible {
        let mut items = Vec::new();
        let mut hidden = 0;
        let readable = |p: &String| access.allows(p) && parts.contains(p);
        for (i, c) in self.chunks.iter().enumerate() {
            if readable(&c.partition) && sources.is_none_or(|s| s.contains(&c.source_id)) {
                if chunk_visible(c, as_of) {
                    items.push(i as u32);
                } else {
                    hidden += 1;
                }
            }
        }
        for (i, f) in self.facts.iter().enumerate() {
            if !readable(&f.partition) {
                continue;
            }
            let shown = match sources {
                None => fact_as_of(f, as_of).is_some(),
                Some(s) => {
                    let mut only = f.clone();
                    only.evidence.retain(|e| s.contains(&e.source_id));
                    if only.evidence.is_empty() {
                        continue;
                    }
                    fact_as_of(&only, as_of).is_some()
                }
            };
            if shown {
                items.push((self.chunks.len() + i) as u32);
            } else {
                hidden += 1;
            }
        }
        let by_id = items.iter().map(|&i| (self.id(i).to_string(), i)).collect();
        Visible {
            items,
            by_id,
            hidden,
        }
    }

    /// BM25 over the visible chunks as a hit shows them (date, speaker and
    /// text, so questions that name a speaker match) and facts' claims.
    pub fn bm25(&self, params: Bm25Params, visible: &Visible) -> Arc<Bm25Index> {
        let mut cache = self.indexes.lock().expect("snapshot lock");
        if let Some((_, idx)) = cache.iter().find(|(k, _)| *k == visible.items) {
            return idx.clone();
        }
        let docs = visible.items.iter().map(|&i| {
            let i = i as usize;
            match self.chunks.get(i) {
                Some(c) => (c.id.clone(), crate::engine::render_chunk(c)),
                None => {
                    let f = &self.facts[i - self.chunks.len()];
                    (f.id.clone(), f.text.clone())
                }
            }
        });
        let idx = Arc::new(Bm25Index::build(params, docs));
        if cache.len() >= INDEXES {
            cache.pop_front();
        }
        cache.push_back((visible.items.clone(), idx.clone()));
        idx
    }
}

/// The most recently used corpus snapshots.
#[derive(Default)]
pub(crate) struct Snapshots(Mutex<VecDeque<(String, Arc<Snapshot>)>>);

impl Snapshots {
    pub fn get(&self, corpus: &str, key: &[(String, i64)]) -> Option<Arc<Snapshot>> {
        let mut s = self.0.lock().expect("snapshots lock");
        let at = s
            .iter()
            .position(|(c, snap)| c == corpus && snap.key == key)?;
        let hit = s.remove(at).expect("present");
        let snap = hit.1.clone();
        s.push_back(hit);
        Some(snap)
    }

    pub fn put(&self, corpus: &str, snap: Arc<Snapshot>) {
        let mut s = self.0.lock().expect("snapshots lock");
        s.retain(|(c, _)| c != corpus);
        if s.len() >= SNAPSHOTS {
            s.pop_front();
        }
        s.push_back((corpus.to_string(), snap));
    }

    pub fn forget(&self, corpus: &str) {
        self.0
            .lock()
            .expect("snapshots lock")
            .retain(|(c, _)| c != corpus);
    }
}
