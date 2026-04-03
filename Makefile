NVCC := nvcc
ARCH ?= 80
ARCHFLAGS := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
BINDIR := bin
BIN := $(BINDIR)/green_ctx_concurrent_test

.PHONY: all clean run

all: $(BIN)

$(BIN): green_ctx_concurrent_test.cu | $(BINDIR)
	$(NVCC) -std=c++20 $(ARCHFLAGS) -o $@ $< -lcuda

$(BINDIR):
	mkdir -p $(BINDIR)

run: $(BIN)
	@echo "Make sure MPS is OFF before running."
	./$(BIN)

clean:
	rm -rf $(BINDIR)
