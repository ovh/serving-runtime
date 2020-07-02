.DEFAULT_GOAL := initialize

WORKDIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

OS :=
ifeq ($(OS),Windows_NT)
	OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OS = linux
	endif
	ifeq ($(UNAME_S),Darwin)
		OS = mac
	endif
endif

.PHONY: initialize
initialize: initialize-${OS}

.PHONY: initialize-mac
initialize-mac:
	curl "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip" -o libtorch-mac-cpu.zip && \
	unzip -q libtorch-mac-cpu.zip && \
	rm -rf libtorch-mac-cpu && mv -vf libtorch libtorch-mac-cpu && rm -v libtorch-mac-cpu.zip
	# Fix OpenMP https://github.com/pytorch/pytorch/issues/38607
	install_name_tool -id @rpath/libiomp5.dylib libtorch-mac-cpu/lib/libiomp5.dylib

.PHONY: initialize-linux
initialize-linux:
	curl "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.1%2Bcpu.zip" -o libtorch-linux-cpu.zip && \
	unzip -q libtorch-linux-cpu.zip && \
	rm -rf libtorch-linux-cpu && mv -vf libtorch libtorch-linux-cpu && rm -v libtorch-linux-cpu.zip
	# Fix OpenMP
	patchelf --set-soname libgomp-7c85b1e2.so.1 libtorch-linux-cpu/lib/libgomp-7c85b1e2.so.1

.PHONY: initialize-windows
initialize-windows:
	curl "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.5.1%2Bcpu.zip" -o libtorch-windows-cpu.zip && \
	unzip -q libtorch-windows-cpu.zip && \
	rm -rf libtorch-windows-cpu && mv -vf libtorch libtorch-windows-cpu && rm -v libtorch-windows-cpu.zip
