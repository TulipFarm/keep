# Changelog

All notable changes to Keep. Versions follow [Semantic Versioning](https://semver.org), and each release is
published as `ghcr.io/tulipfarm/keep:<version>`.

## [0.1.1](https://github.com/TulipFarm/keep/releases/tag/v0.1.1) - 2026-11-26

### Performance

- Bound ingest work, reuse stored vectors and cap ONNX CPU ([#14](https://github.com/TulipFarm/keep/pull/14))

### Documentation

- **readme:** Add release and container badges, point to releases ([#13](https://github.com/TulipFarm/keep/pull/13))

## [0.1.0](https://github.com/TulipFarm/keep/releases/tag/v0.1.0) - 2026-11-25

### Features

- Host integration: source allowlists, revisions, model extraction, image release ([#2](https://github.com/TulipFarm/keep/pull/2))
- **memory:** Place /you/<slug>.md files in the You category ([#5](https://github.com/TulipFarm/keep/pull/5))

### Bug fixes

- **knowledge:** Keep readiness answering during a long source purge ([#7](https://github.com/TulipFarm/keep/pull/7))
- **release:** Match the exact repository remote ([#12](https://github.com/TulipFarm/keep/pull/12))

### Performance

- **knowledge:** Skip finished jobs when counting on ingest ([#4](https://github.com/TulipFarm/keep/pull/4))
- Budget ingest jobs, cache local embeddings, clamp ONNX threads ([#9](https://github.com/TulipFarm/keep/pull/9))

### Documentation

- Make agent install via install.md the primary install path ([#6](https://github.com/TulipFarm/keep/pull/6))

### Build and CI

- Release semver versions with scripts/release.sh ([#11](https://github.com/TulipFarm/keep/pull/11))

### Dependencies

- Upgrade all dependencies ([#10](https://github.com/TulipFarm/keep/pull/10))

### Other changes

- Open-source memory and knowledge for AI agents
- Add the Keep landing page ([#1](https://github.com/TulipFarm/keep/pull/1))
- Add banner, badges and project links ([#3](https://github.com/TulipFarm/keep/pull/3))
- **rust:** Upgrade to Rust 1.100.0 ([#8](https://github.com/TulipFarm/keep/pull/8))

