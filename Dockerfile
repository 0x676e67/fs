FROM rust:1.79 as builder

WORKDIR /src

COPY Cargo.toml Cargo.toml
COPY src src
RUN cargo build --release

FROM ubuntu:noble-20231214

LABEL org.opencontainers.image.source = "https://github.com/0x676e67/fc"
COPY --from=builder /src/target/release/fc /bin/fc
ENTRYPOINT [ "/bin/fc"]
