.PHONY: build
build:
	cd ../triton && \
	TRITON_PLUGIN_DIRS=../triton_tvm \
	TRITON_BUILD_WITH_CLANG_LLD=true \
	TRITON_BUILD_WITH_CCACHE=true \
	python3 -m pip install -e . --no-build-isolation -v

.PHONY: test
test:
	python example/add.py
	python example/softmax.py