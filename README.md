# Green Context Concurrent Execution on A100

Two green contexts, two kernels, mapped host memory flag. If they run concurrently: PASS. If serialized: deadlock (5s timeout).

## Build & Run

```bash
make clean && make && make run
```

MPS must be OFF.

## The Test

1. Create two CUDA Green Contexts with disjoint SM partitions (4 SMs each out of 108)
2. Launch a persistent kernel in ctx[0] that spins on a flag
3. Launch a signal kernel in ctx[1] that sets the flag
4. If both run concurrently: signal sets flag → persistent exits → PASS
5. If serialized: persistent blocks signal from ever launching → deadlock → FAIL (5s timeout)

Both kernels use `cudaHostAllocMapped` memory. Each is launched from a separate OS thread on a `cudaStreamNonBlocking` stream.

## From NVIDIA Docs

[CUDA Programming Guide §4.6: Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html):

> "Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees. This is due to other resources (like HW connections, see `CUDA_DEVICE_MAX_CONNECTIONS`) that could cause a dependency."

[NVIDIA Forum: Green Context SM Allocation](https://forums.developer.nvidia.com/t/green-context-sm-allocation-not-affecting-kernel-runtime/331825): Concurrent green context execution confirmed working on A40 (CUDA 12.8) with independent kernels, but not on Jetson Orin Nano (CUDA 12.6, known bug fixed in 12.8).

## Implications

Any system requiring two concurrent kernels with forward progress guarantees (e.g., a persistent service kernel + an application kernel) cannot rely on green contexts alone. MPS (`nvidia-cuda-mps-control`) provides guaranteed concurrent execution between contexts. Green contexts do not.

## Hardware

- NVIDIA A100-PCIE-40GB, 108 SMs
- CUDA 13.0, driver 580.65.06
