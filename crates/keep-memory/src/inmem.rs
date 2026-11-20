//! In-memory ledger and journal for tests, benchmarks and single-process use.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::error::Result;
use crate::ledger::{ErasureJournal, LedgerTx, MemoryLedger};
use crate::model::{ErasureRecord, FileRecord, Receipt, Revision, TreeRecord};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InMemoryState {
    trees: BTreeMap<String, TreeRecord>,
    files: BTreeMap<String, FileRecord>,
    revisions: BTreeMap<String, Revision>,
    receipts: BTreeMap<(String, String, String), Receipt>,
}

impl InMemoryState {
    /// Every revision currently stored, for assertions in tests.
    pub fn all_revisions(&self) -> impl Iterator<Item = &Revision> {
        self.revisions.values()
    }

    pub fn all_files(&self) -> impl Iterator<Item = &FileRecord> {
        self.files.values()
    }

    pub fn all_receipts(&self) -> impl Iterator<Item = &Receipt> {
        self.receipts.values()
    }
}

/// A ledger whose transactions serialise on one lock. A transaction works on
/// a copy of the state and swaps it in on commit.
#[derive(Clone, Default)]
pub struct InMemoryLedger {
    state: Arc<Mutex<InMemoryState>>,
}

impl InMemoryLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of the whole state, like a backup.
    pub async fn snapshot(&self) -> InMemoryState {
        self.state.lock().await.clone()
    }

    /// Replace the whole state, like restoring a backup.
    pub async fn restore(&self, state: InMemoryState) {
        *self.state.lock().await = state;
    }
}

struct InMemoryTx {
    guard: OwnedMutexGuard<InMemoryState>,
    work: InMemoryState,
}

#[async_trait]
impl MemoryLedger for InMemoryLedger {
    async fn begin(&self) -> Result<Box<dyn LedgerTx + '_>> {
        let guard = self.state.clone().lock_owned().await;
        let work = guard.clone();
        Ok(Box::new(InMemoryTx { guard, work }))
    }
}

#[async_trait]
impl LedgerTx for InMemoryTx {
    async fn tree(&mut self, tree_id: &str) -> Result<Option<TreeRecord>> {
        Ok(self.work.trees.get(tree_id).cloned())
    }

    async fn put_tree(&mut self, tree: &TreeRecord) -> Result<()> {
        self.work.trees.insert(tree.tree_id.clone(), tree.clone());
        Ok(())
    }

    async fn files(&mut self, tree_id: &str, include_trashed: bool) -> Result<Vec<FileRecord>> {
        let mut out: Vec<FileRecord> = self
            .work
            .files
            .values()
            .filter(|f| f.tree_id == tree_id && (include_trashed || f.is_live()))
            .cloned()
            .collect();
        out.sort_by(|a, b| (&a.meta.path, &a.memory_id).cmp(&(&b.meta.path, &b.memory_id)));
        Ok(out)
    }

    async fn file(&mut self, memory_id: &str) -> Result<Option<FileRecord>> {
        Ok(self.work.files.get(memory_id).cloned())
    }

    async fn put_file(&mut self, file: &FileRecord) -> Result<()> {
        self.work.files.insert(file.memory_id.clone(), file.clone());
        Ok(())
    }

    async fn purge_file(&mut self, memory_id: &str) -> Result<()> {
        self.work.files.remove(memory_id);
        self.work.revisions.retain(|_, r| r.memory_id != memory_id);
        Ok(())
    }

    async fn put_revision(&mut self, revision: &Revision) -> Result<()> {
        self.work
            .revisions
            .insert(revision.revision_id.clone(), revision.clone());
        Ok(())
    }

    async fn revision(&mut self, revision_id: &str) -> Result<Option<Revision>> {
        Ok(self.work.revisions.get(revision_id).cloned())
    }

    async fn revisions(&mut self, memory_id: &str) -> Result<Vec<Revision>> {
        let mut out: Vec<Revision> = self
            .work
            .revisions
            .values()
            .filter(|r| r.memory_id == memory_id)
            .cloned()
            .collect();
        out.sort_by_key(|r| r.seq);
        Ok(out)
    }

    async fn delete_revision(&mut self, revision_id: &str) -> Result<()> {
        self.work.revisions.remove(revision_id);
        Ok(())
    }

    async fn receipt(
        &mut self,
        tree_id: &str,
        principal: &str,
        key: &str,
    ) -> Result<Option<Receipt>> {
        Ok(self
            .work
            .receipts
            .get(&(tree_id.to_owned(), principal.to_owned(), key.to_owned()))
            .cloned())
    }

    async fn put_receipt(&mut self, receipt: &Receipt) -> Result<()> {
        self.work.receipts.insert(
            (
                receipt.tree_id.clone(),
                receipt.principal.clone(),
                receipt.key.clone(),
            ),
            receipt.clone(),
        );
        Ok(())
    }

    async fn receipts_for_file(&mut self, memory_id: &str) -> Result<Vec<Receipt>> {
        Ok(self
            .work
            .receipts
            .values()
            .filter(|r| r.memory_id.as_deref() == Some(memory_id))
            .cloned()
            .collect())
    }

    async fn commit(self: Box<Self>) -> Result<()> {
        let InMemoryTx { mut guard, work } = *self;
        *guard = work;
        Ok(())
    }
}

/// A journal kept in process memory. It is independent of an
/// [`InMemoryLedger`] in the sense that restoring the ledger leaves it intact.
#[derive(Clone, Default)]
pub struct InMemoryJournal {
    records: Arc<Mutex<Vec<ErasureRecord>>>,
}

impl InMemoryJournal {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ErasureJournal for InMemoryJournal {
    async fn append(&self, mut record: ErasureRecord) -> Result<u64> {
        let mut records = self.records.lock().await;
        record.seq = records.len() as u64 + 1;
        let seq = record.seq;
        records.push(record);
        Ok(seq)
    }

    async fn records(&self, tree_id: &str) -> Result<Vec<ErasureRecord>> {
        Ok(self
            .records
            .lock()
            .await
            .iter()
            .filter(|r| r.tree_id == tree_id)
            .cloned()
            .collect())
    }

    fn independent(&self) -> bool {
        true
    }
}
