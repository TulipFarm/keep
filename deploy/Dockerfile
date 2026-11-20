# Keep server image: `keep serve` on a minimal Debian base, as a non-root
# user. Base images are pinned by multi-architecture digest.
#
#   docker build -f deploy/Dockerfile -t keep .
#
# Build with the on-device embedder. The pinned model is downloaded and
# checked against its SHA-256 digests at build time and baked into the
# image (KEEP_MODEL_CACHE), so the server never downloads it at runtime:
#   docker build -f deploy/Dockerfile --build-arg KEEP_FEATURES=local-embeddings -t keep .

FROM rust:1.99.1-slim-trixie@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7 AS build
ARG KEEP_FEATURES=""
ARG TARGETPLATFORM
# The ONNX Runtime the local embedder links statically is C++ and needs
# the GCC 14 libstdc++ of Debian 13 (trixie).
RUN case ",$KEEP_FEATURES," in \
        *,local-embeddings,*) apt-get update \
            && apt-get install -y --no-install-recommends g++ \
            && rm -rf /var/lib/apt/lists/* ;; \
    esac
WORKDIR /src
COPY . .
RUN --mount=type=cache,id=keep-cargo-registry,sharing=locked,target=/usr/local/cargo/registry \
    --mount=type=cache,id=keep-target-$TARGETPLATFORM,sharing=locked,target=/src/target \
    cargo build --release --locked -p keep-server --features "$KEEP_FEATURES" \
    && install -m 0755 target/release/keep /usr/local/bin/keep
RUN mkdir -p /opt/keep/models \
    && case ",$KEEP_FEATURES," in \
        *,local-embeddings,*) keep fetch-model --dir /opt/keep/models ;; \
    esac

FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /var/lib/keep --create-home keep
COPY --from=build /usr/local/bin/keep /usr/local/bin/keep
COPY --from=build /opt/keep/models /opt/keep/models
COPY deploy/entrypoint.sh /usr/local/bin/keep-entrypoint
USER keep
ENV KEEP_DATA_DIR=/var/lib/keep \
    KEEP_MODEL_CACHE=/opt/keep/models
EXPOSE 7357
ENTRYPOINT ["keep-entrypoint"]
CMD ["serve"]
