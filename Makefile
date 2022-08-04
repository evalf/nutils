develop: build
	cp --reflink=auto target/debug/libnutils.so nutils/_rust.so

develop-release: build-release
	cp --reflink=auto target/release/libnutils.so nutils/_rust.so

build:
	cargo build

build-release:
	cargo build --release

test-rust:
	cargo test

docs-python: develop-release
	python3 -m sphinx -n -W --keep-going -E -D html_theme=sphinx_rtd_theme docs target/doc/nutils-python

.PHONY: build
