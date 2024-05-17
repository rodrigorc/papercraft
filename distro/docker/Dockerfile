FROM ubuntu:20.04
WORKDIR /app
RUN ln -sf /usr/share/zoneinfo/Europe/Madrid /etc/localtime
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install curl gcc pkg-config xz-utils fuse libclang-dev file libfreetype6-dev
RUN apt -y install g++
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.rs
# Rust 1.76
RUN sh rustup.rs -y --profile minimal -c rustfmt
copy g++ /usr/local/bin/g++
copy g++ /usr/local/bin/c++
COPY linuxdeploy ./
COPY makeappimage ./
COPY apprun ./
