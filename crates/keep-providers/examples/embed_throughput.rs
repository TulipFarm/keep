//! Local embedding throughput on synthetic passages of varied length.
//! Offline: it only loads the cached pinned model.
//!
//! `KEEP_MODEL_CACHE=... KEEP_ONNX_DEVICE=cpu|auto KEEP_ONNX_THREADS=n \
//!  KEEP_ONNX_SESSIONS=n cargo run --release -p keep-providers \
//!  --features local-embeddings --example embed_throughput -- [passages] [call]`
//!
//! `call` is how many texts each `embed` call carries. On a device other
//! than the CPU it also prints how far its vectors are from the CPU's.

use std::path::PathBuf;
use std::time::Instant;

use keep_providers::Embedder;
use keep_providers::local_embed::{LocalDevice, LocalEmbedConfig, LocalEmbedder};

pub fn passages(n: usize) -> Vec<String> {
    let words = [
        "the", "meeting", "moved", "to", "tuesday", "and", "we", "talked", "about", "the",
        "garden", "trip", "my", "sister", "booked", "flights", "for", "march", "budget",
    ];
    (0..n)
        .map(|i| {
            (0..20 + (i * 37) % 400)
                .map(|j| words[(i * 7 + j * 13) % words.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(Ok(256), |a| a.parse())?;
    let call: usize = args.next().map_or(Ok(64), |a| a.parse())?;
    let cache = std::env::var_os("KEEP_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".keep-models"));
    let config = LocalEmbedConfig::bge_small_en_v1_5(cache.clone()).with_env_overrides()?;
    let (device, (sessions, threads)) = (config.device, config.parallelism());
    let started = Instant::now();
    let e = LocalEmbedder::load(config).await?;
    let load = started.elapsed();
    let texts = passages(n);
    e.embed(&texts[..call.min(n)]).await?;
    let started = Instant::now();
    for b in texts.chunks(call) {
        e.embed(b).await?;
    }
    let secs = started.elapsed().as_secs_f64();
    println!(
        "device={device:?} sessions={sessions} threads={threads} call={call} passages={n} load={:.2}s embed={secs:.2}s rate={:.1}/s",
        load.as_secs_f64(),
        n as f64 / secs
    );
    if device != LocalDevice::Cpu {
        let sample = &texts[..64.min(n)];
        let cpu = LocalEmbedder::load(LocalEmbedConfig::bge_small_en_v1_5(cache)).await?;
        let (a, b) = (e.embed(sample).await?, cpu.embed(sample).await?);
        let delta = a
            .vectors
            .iter()
            .flatten()
            .zip(b.vectors.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!(
            "max |difference| from the cpu over {} vectors: {delta:e}",
            sample.len()
        );
    }
    Ok(())
}
