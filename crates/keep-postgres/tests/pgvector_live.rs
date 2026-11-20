//! Opt-in test against a real Postgres with pgvector.
//! Set KEEP_TEST_POSTGRES_URL (see compose.yaml) to run it.

use keep_postgres::PgVectorIndex;
use keep_store::{VectorIndex, VectorPoint};

#[tokio::test]
async fn pgvector_round_trip_with_namespace_isolation() {
    let Ok(url) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let idx = PgVectorIndex::connect(&url).await.unwrap();
    let col = format!("test_{}", std::process::id());
    idx.ensure_collection(&col, 3).await.unwrap();
    idx.upsert(
        &col,
        "conv-a",
        &[
            VectorPoint {
                id: "a1".into(),
                vector: vec![1.0, 0.0, 0.0],
                tags: vec!["s1".into()],
            },
            VectorPoint {
                id: "a2".into(),
                vector: vec![0.0, 1.0, 0.0],
                tags: vec!["s2".into()],
            },
        ],
    )
    .await
    .unwrap();
    idx.upsert(
        &col,
        "conv-b",
        &[VectorPoint {
            id: "b1".into(),
            vector: vec![1.0, 0.0, 0.0],
            tags: vec![],
        }],
    )
    .await
    .unwrap();
    let hits = idx
        .search(&col, "conv-a", &[0.9, 0.1, 0.0], 5)
        .await
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["a1", "a2"]
    );
    assert!(hits[0].score > hits[1].score);
    // The tag filter applies before the limit: the farther point is found.
    let tagged = idx
        .search_tagged(&col, "conv-a", &[0.9, 0.1, 0.0], 1, &["s2".into()])
        .await
        .unwrap();
    assert_eq!(
        tagged.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
        ["a2"]
    );
    assert!(
        idx.search_tagged(&col, "conv-b", &[1.0, 0.0, 0.0], 5, &["s1".into()])
            .await
            .unwrap()
            .is_empty()
    );
    // Re-upserting replaces the tags.
    idx.upsert(
        &col,
        "conv-a",
        &[VectorPoint {
            id: "a1".into(),
            vector: vec![1.0, 0.0, 0.0],
            tags: vec!["s2".into()],
        }],
    )
    .await
    .unwrap();
    assert!(
        idx.search_tagged(&col, "conv-a", &[1.0, 0.0, 0.0], 5, &["s1".into()])
            .await
            .unwrap()
            .is_empty()
    );
    assert!(idx.ensure_collection(&col, 4).await.is_err());
    idx.delete_namespace(&col, "conv-a").await.unwrap();
    assert!(
        idx.search(&col, "conv-a", &[1.0, 0.0, 0.0], 5)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        idx.search(&col, "conv-b", &[1.0, 0.0, 0.0], 5)
            .await
            .unwrap()
            .len(),
        1
    );
}
