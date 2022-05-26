FROM ubuntu:18.04
WORKDIR /app
RUN apt -y update
RUN apt -y install curl gcc pkg-config xz-utils libgtk-3-dev fuse
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.rs
# Ubuntu 18.04 does not ship cairo-1.16 that we need, so compile it locally
COPY build_cairo ./
RUN ./build_cairo
RUN sh rustup.rs -y --profile minimal
COPY linuxdeploy ./
COPY makeappimage ./
COPY apprun ./
