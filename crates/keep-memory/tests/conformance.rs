use std::sync::Arc;

use keep_core::memory::MemoryLimits;
use keep_memory::conformance::{self, Backend, TREE, fixture};
use keep_memory::service::{CreateFile, Edit, ForgetSelector, OpCtx};
use keep_memory::{InMemoryJournal, InMemoryLedger, MemoryError};

#[tokio::test]
async fn in_memory_ledger_passes_the_conformance_suite() {
    conformance::run(|| async {
        Backend {
            ledger: Arc::new(InMemoryLedger::new()),
            journal: Arc::new(InMemoryJournal::new()),
        }
    })
    .await;
}

/// Restoring an old backup brings forgotten text back; replaying the
/// erasure journal removes it again.
#[tokio::test]
async fn journal_replay_cleans_a_restored_backup() {
    let ledger = InMemoryLedger::new();
    let f = fixture(
        Backend {
            ledger: Arc::new(ledger.clone()),
            journal: Arc::new(InMemoryJournal::new()),
        },
        MemoryLimits::default(),
    )
    .await;
    let ctx = OpCtx::new("user:sam").with_key("k1");
    let a = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/a.md", "# A\nsecret-xyz\n# B\nfine\n"),
        )
        .await
        .expect("a");
    let plain = OpCtx::new("user:sam");
    let b = f
        .svc
        .create(
            &plain,
            TREE,
            &CreateFile::new("/topics/b.md", "secret-xyz early\n"),
        )
        .await
        .expect("b");
    f.svc
        .edit(
            &plain,
            TREE,
            &b.memory_id,
            &b.head,
            &Edit::Write {
                text: "later\n".into(),
            },
        )
        .await
        .expect("edit b");
    let backup = ledger.snapshot().await;

    f.svc
        .forget(
            &plain,
            TREE,
            &ForgetSelector::Text {
                memory_id: None,
                text: "secret-xyz".into(),
            },
        )
        .await
        .expect("forget");
    ledger.restore(backup).await;
    assert!(
        f.svc
            .view(TREE, &a.memory_id)
            .await
            .expect("view")
            .text()
            .contains("secret-xyz")
    );

    let replay = f.svc.replay_erasure_journal(TREE).await.expect("replay");
    assert!(replay.revisions_purged >= 2);
    let state = ledger.snapshot().await;
    assert!(
        state
            .all_revisions()
            .all(|r| !r.text().contains("secret-xyz"))
    );
    // a's head was erased, so the conservative replay purged the file; b
    // keeps its clean head.
    assert!(matches!(
        f.svc.view(TREE, &a.memory_id).await,
        Err(MemoryError::NotFound { .. })
    ));
    assert_eq!(
        f.svc.view(TREE, &b.memory_id).await.expect("b").text(),
        "later\n"
    );
    assert!(state.all_receipts().all(|r| r.digest.is_none()));
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

/// Concurrent writers to one file: exactly one compare-and-swap wins.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_have_one_winner() {
    let f = Arc::new(
        fixture(
            Backend {
                ledger: Arc::new(InMemoryLedger::new()),
                journal: Arc::new(InMemoryJournal::new()),
            },
            MemoryLimits::default(),
        )
        .await,
    );
    let ctx = OpCtx::new("user:sam");
    let c = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/race.md", "start\n"))
        .await
        .expect("create");
    let mut tasks = Vec::new();
    for i in 0..8 {
        let f = f.clone();
        let (id, head) = (c.memory_id.clone(), c.head.clone());
        tasks.push(tokio::spawn(async move {
            f.svc
                .edit(
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
}
