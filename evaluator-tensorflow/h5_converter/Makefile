.DEFAULT_GOAL := build

WORKDIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: build
build:
	pip3 install -r requirements.txt
	bash -c "pyinstaller --clean --additional-hooks-dir=./hooks -F h5_converter.py --log-level=INFO --add-data $$(python3 -c 'import os;import astor;print(os.path.dirname(astor.__file__))')/VERSION:astor/"
