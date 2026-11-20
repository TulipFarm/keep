//! Storage contracts.
//!
//! A [`MemoryLedger`] hands out transactions. Every service operation runs
//! inside one transaction: reads take the tree lock first (the Postgres
//! ledger uses `SELECT … FOR UPDATE`), writes are invisible until
//! [`LedgerTx::commit`], and dropping a transaction without committing
//! discards it.
//!
//! The [`ErasureJournal`] is deliberately separate. It records content-free
//! erasure decisions *before* the ledger commits them, so that a database
//! restored from an older backup can be cleaned again by replaying it.

use async_trait::async_trait;

use crate::error::Result;
use crate::model::{ErasureRecord, FileRecord, Receipt, Revision, TreeRecord};

#[async_trait]
pub trait MemoryLedger: Send + Sync {
    async fn begin(&self) -> Result<Box<dyn LedgerTx + '_>>;
}

#[async_trait]
pub trait LedgerTx: Send {
    /// Read a tree and lock it for the rest of the transaction.
    async fn tree(&mut self, tree_id: &str) -> Result<Option<TreeRecord>>;
    async fn put_tree(&mut self, tree: &TreeRecord) -> Result<()>;

    /// Files of a tree ordered by path; trashed files only when asked.
    async fn files(&mut self, tree_id: &str, include_trashed: bool) -> Result<Vec<FileRecord>>;
    async fn file(&mut self, memory_id: &str) -> Result<Option<FileRecord>>;
    async fn put_file(&mut self, file: &FileRecord) -> Result<()>;
    /// Remove a file and every revision of it.
    async fn purge_file(&mut self, memory_id: &str) -> Result<()>;

    async fn put_revision(&mut self, revision: &Revision) -> Result<()>;
    async fn revision(&mut self, revision_id: &str) -> Result<Option<Revision>>;
    /// All retained revisions of a file, oldest first.
    async fn revisions(&mut self, memory_id: &str) -> Result<Vec<Revision>>;
    async fn delete_revision(&mut self, revision_id: &str) -> Result<()>;

    async fn receipt(
        &mut self,
        tree_id: &str,
        principal: &str,
        key: &str,
    ) -> Result<Option<Receipt>>;
    async fn put_receipt(&mut self, receipt: &Receipt) -> Result<()>;
    async fn receipts_for_file(&mut self, memory_id: &str) -> Result<Vec<Receipt>>;

    async fn commit(self: Box<Self>) -> Result<()>;
}

#[async_trait]
pub trait ErasureJournal: Send + Sync {
    /// Append a record and return its sequence number.
    async fn append(&self, record: ErasureRecord) -> Result<u64>;
    /// Records for a tree, in append order.
    async fn records(&self, tree_id: &str) -> Result<Vec<ErasureRecord>>;
    /// Whether the journal survives independently of the ledger. Reported on
    /// forget receipts so callers know the recovery guarantee they got.
    fn independent(&self) -> bool;
}
