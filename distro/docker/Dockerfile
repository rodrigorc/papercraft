FROM ubuntu:18.04
WORKDIR /app
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install curl gcc pkg-config xz-utils fuse libclang-dev file libfreetype6-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.rs
# Rust 1.76
RUN sh rustup.rs -y --profile minimal -c rustfmt
COPY linuxdeploy ./
COPY makeappimage ./
COPY apprun ./
