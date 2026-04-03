NVCC := nvcc
ARCH ?= 80
ARCHFLAGS := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
BINDIR := bin
BIN := $(BINDIR)/green_ctx_test

.PHONY: all clean run

all: $(BIN)

$(BIN): green_ctx_concurrent_test.cu | $(BINDIR)
	$(NVCC) -std=c++20 $(ARCHFLAGS) -o $@ $< -lcuda

$(BINDIR):
	mkdir -p $(BINDIR)

run: $(BIN)
	@echo "MPS must be OFF. Run multiple times — result is non-deterministic."
	@./$(BIN)

clean:
	rm -rf $(BINDIR)
