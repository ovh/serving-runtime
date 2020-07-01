.DEFAULT_GOAL := usage

WORKDIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: initialize-usage
usage:
	echo 'Choose a goal between initialize-mac, initialize-linux and initialize-windows'

.PHONY: initialize-mac
initialize-mac:
	curl "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip" -o libtorch-mac-cpu.zip && \
	unzip -q libtorch-mac-cpu.zip && \
	rm -rf libtorch-mac-cpu && mv -vf libtorch libtorch-mac-cpu && rm -v libtorch-mac-cpu.zip
	# Fix https://github.com/pytorch/pytorch/issues/38607
	install_name_tool -id @rpath/libiomp5.dylib libtorch-mac-cpu/lib/libiomp5.dylib

.PHONY: initialize-linux
initialize-linux:
	curl "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip" -o libtorch-linux-cpu.zip && \
	unzip -q libtorch-linux-cpu.zip && \
	rm -vrf libtorch-linux-cpu && mv -vf libtorch libtorch-linux-cpu && rm -v libtorch-linux-cpu.zip

.PHONY: initialize-windows
initialize-windows:
	curl "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.5.1%2Bcpu.zip" -o libtorch-windows-cpu.zip && \
	unzip -q libtorch-windows-cpu.zip && \
	rm -vrf libtorch-windows-cpu && mv -vf libtorch libtorch-windows-cpu && rm -v libtorch-windows-cpu.zip
