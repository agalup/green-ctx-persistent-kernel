// Can two kernels in different CUDA Green Contexts run concurrently on A100?
//
// Test: persistent kernel spins on a flag in ctx[0].
//       Signal kernel sets the flag from ctx[1].
//       If concurrent: both complete. If serialized: deadlock.
//
// Result: NON-DETERMINISTIC. Same test, same GPU — sometimes works, sometimes deadlocks.
//
// A100-PCIE-40GB, CUDA 13.0, driver 580.65.06.

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thread>
#include <chrono>

#define CK(x) do{CUresult _r=(x);if(_r!=CUDA_SUCCESS){const char*_s;cuGetErrorString(_r,&_s);printf("FAIL line %d: %s\n",__LINE__,_s);exit(1);}}while(0)

__global__ void spin(volatile int* flag, volatile int* out) {
    if (threadIdx.x == 0) {
        while (flag[0] == 0) __nanosleep(100);
        out[0] = 1;
        __threadfence_system();
    }
}

__global__ void signal(volatile int* flag) {
    if (threadIdx.x == 0) {
        flag[0] = 1;
        __threadfence_system();
    }
}

int main() {
    CUdevice dev;
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));

    // Create two green contexts with 4 SMs each
    CUdevResource devr{};
    CK(cuDeviceGetDevResource(dev, &devr, CU_DEV_RESOURCE_TYPE_SM));

    CUdevResource parts[2]{};
    unsigned n = 2;
    CK(cuDevSmResourceSplitByCount(parts, &n, &devr, nullptr, 0, 4));

    CUgreenCtx g[2];
    CUcontext ctx[2];
    for (int i = 0; i < 2; i++) {
        CUdevResourceDesc d;
        CK(cuDevResourceGenerateDesc(&d, &parts[i], 1));
        CK(cuGreenCtxCreate(&g[i], d, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        CK(cuCtxFromGreenCtx(&ctx[i], g[i]));
    }

    // Shared flag and output in mapped host memory
    volatile int *flag, *out;
    cudaHostAlloc((void**)&flag, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable);
    cudaHostAlloc((void**)&out, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable);
    *flag = 0;
    *out = 0;

    printf("Green contexts: %u SMs + %u SMs (of %u total)\n",
        parts[0].sm.smCount, parts[1].sm.smCount, devr.sm.smCount);
    printf("Launching persistent kernel in ctx[0], signal kernel in ctx[1]...\n");

    // Launch persistent kernel in ctx[0] (separate thread)
    std::thread t0([&]{
        CK(cuCtxPushCurrent(ctx[0]));
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        spin<<<1, 32, 0, s>>>(flag, out);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
        CK(cuCtxPopCurrent(nullptr));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Launch signal kernel in ctx[1] (separate thread)
    std::thread t1([&]{
        CK(cuCtxPushCurrent(ctx[1]));
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        signal<<<1, 32, 0, s>>>(flag);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
        CK(cuCtxPopCurrent(nullptr));
    });

    // Wait with timeout
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (*out == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (*out == 1) {
        printf("PASS: kernels ran concurrently\n");
    } else {
        printf("FAIL: deadlock (signal kernel never ran)\n");
        printf("\nCUDA docs: \"Even if the green contexts have disjoint SM partitions,\n"
               "it is not guaranteed that the kernels launched in them will run\n"
               "concurrently or have forward progress guarantees.\"\n");
        *flag = 1; // force-stop
    }

    t0.join();
    t1.join();
    cudaFreeHost((void*)flag);
    cudaFreeHost((void*)out);
    return 0;
}
