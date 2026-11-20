# Keep server image: `keep serve` on a minimal Debian base, as a non-root
# user. Base images are pinned by multi-architecture digest.
#
#   docker build -f deploy/Dockerfile -t keep .
#
# Build with the on-device embedder (downloads the model on first start):
#   docker build -f deploy/Dockerfile --build-arg KEEP_FEATURES=local-embeddings -t keep .

FROM rust:1.99.1-slim-bookworm@sha256:f2c004e3d2d8b547e8cf7ffbc29205385f4e71d23e53e4e383065db5e9f822cd AS build
ARG KEEP_FEATURES=""
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p keep-server --features "$KEEP_FEATURES" \
    && install -m 0755 target/release/keep /usr/local/bin/keep

FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /var/lib/keep --create-home keep
COPY --from=build /usr/local/bin/keep /usr/local/bin/keep
COPY deploy/entrypoint.sh /usr/local/bin/keep-entrypoint
USER keep
ENV KEEP_DATA_DIR=/var/lib/keep
EXPOSE 7357
ENTRYPOINT ["keep-entrypoint"]
CMD ["serve"]
