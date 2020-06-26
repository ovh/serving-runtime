.DEFAULT_GOAL := build

WORKDIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: build
build:
	cargo build --release
