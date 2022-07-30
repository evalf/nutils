develop: build
	cp --reflink=auto target/debug/libnutils.so nutils/_rust.so

develop-release: build-release
	cp --reflink=auto target/release/libnutils.so nutils/_rust.so

build:
	cargo build

build-release:
	cargo build --release

bench:
	cargo +nightly bench --features bench

bench-python-compare:
	python3 -m pytest benches/ --benchmark-compare --benchmark-group-by=name

test-rust:
	cargo test

docs-python: develop-release
	python3 -m sphinx -n -W --keep-going -E -D html_theme=sphinx_rtd_theme docs build/sphinx/html

.PHONY: build
