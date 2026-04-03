// Green context concurrent execution test.
// Tests multiple launch strategies to find one that works.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thread>
#include <chrono>

#define CHECK_CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(r, &s); \
    printf("CUDA driver error %d (%s) at %s:%d\n", r, s, __FILE__, __LINE__); \
    exit(1); } } while(0)

#define CHECK_RT(x) do { cudaError_t r = (x); if (r != cudaSuccess) { \
    printf("CUDA runtime error %d (%s) at %s:%d\n", r, cudaGetErrorString(r), __FILE__, __LINE__); \
    exit(1); } } while(0)

__global__ void persistent_kernel(volatile int* flag, volatile int* output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (flag[0] == 0) { __nanosleep(100); }
        output[0] = 42;
        __threadfence_system();
    }
}

__global__ void short_kernel(volatile int* flag, volatile int* output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = 99;
        __threadfence_system();
        flag[0] = 1;
        __threadfence_system();
    }
}

__global__ void independent_kernel(volatile int* output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = 77;
        __threadfence_system();
    }
}

struct GreenCtxPair {
    CUgreenCtx green[2];
    CUcontext ctx[2];
    CUdevResource res[2];
};

bool create_green_contexts(CUdevice device, int minSm, GreenCtxPair& p)
{
    CUdevResource dev{};
    if (cuDeviceGetDevResource(device, &dev, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS) return false;
    unsigned int n = 0;
    if (cuDevSmResourceSplitByCount(nullptr, &n, &dev, nullptr, 0, minSm) != CUDA_SUCCESS || n < 2) return false;
    unsigned int actual = 2;
    if (cuDevSmResourceSplitByCount(p.res, &actual, &dev, nullptr, 0, minSm) != CUDA_SUCCESS || actual < 2) return false;
    for (int i = 0; i < 2; i++) {
        CUdevResourceDesc desc;
        CHECK_CU(cuDevResourceGenerateDesc(&desc, &p.res[i], 1));
        CHECK_CU(cuGreenCtxCreate(&p.green[i], desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
        CHECK_CU(cuCtxFromGreenCtx(&p.ctx[i], p.green[i]));
    }
    return true;
}

bool wait_for(volatile int* ptr, int val, int ms)
{
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (*ptr != val) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

void reset(volatile int* f, volatile int* o0, volatile int* o1) { *f=0; *o0=0; *o1=0; }

void test_push_pop(GreenCtxPair& p, volatile int* flag, volatile int* o0, volatile int* o1)
{
    printf("\n--- Test 1: push/pop, non-blocking streams ---\n");
    reset(flag, o0, o1);

    CHECK_CU(cuCtxPushCurrent(p.ctx[0]));
    cudaStream_t s0; CHECK_RT(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    persistent_kernel<<<1,32,0,s0>>>(flag, o0);
    CHECK_RT(cudaGetLastError());
    CHECK_CU(cuCtxPopCurrent(nullptr));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    CHECK_CU(cuCtxPushCurrent(p.ctx[1]));
    cudaStream_t s1; CHECK_RT(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    short_kernel<<<1,32,0,s1>>>(flag, o1);
    CHECK_RT(cudaGetLastError());
    CHECK_CU(cuCtxPopCurrent(nullptr));

    if (wait_for(o0, 42, 3000)) printf("  PASS\n");
    else { printf("  FAIL (timeout)\n"); *flag=1; }

    CHECK_CU(cuCtxPushCurrent(p.ctx[0])); cudaStreamSynchronize(s0); cudaStreamDestroy(s0); CHECK_CU(cuCtxPopCurrent(nullptr));
    CHECK_CU(cuCtxPushCurrent(p.ctx[1])); cudaStreamSynchronize(s1); cudaStreamDestroy(s1); CHECK_CU(cuCtxPopCurrent(nullptr));
}

void test_nested(GreenCtxPair& p, volatile int* flag, volatile int* o0, volatile int* o1)
{
    printf("\n--- Test 2: nested push (ctx1 on top of ctx0) ---\n");
    reset(flag, o0, o1);

    CHECK_CU(cuCtxPushCurrent(p.ctx[0]));
    cudaStream_t s0; CHECK_RT(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    persistent_kernel<<<1,32,0,s0>>>(flag, o0);
    CHECK_RT(cudaGetLastError());

    CHECK_CU(cuCtxPushCurrent(p.ctx[1]));
    cudaStream_t s1; CHECK_RT(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    short_kernel<<<1,32,0,s1>>>(flag, o1);
    CHECK_RT(cudaGetLastError());
    CHECK_CU(cuCtxPopCurrent(nullptr));

    if (wait_for(o0, 42, 3000)) printf("  PASS\n");
    else { printf("  FAIL (timeout)\n"); *flag=1; }

    cudaStreamSynchronize(s0); cudaStreamDestroy(s0);
    CHECK_CU(cuCtxPushCurrent(p.ctx[1])); cudaStreamDestroy(s1); CHECK_CU(cuCtxPopCurrent(nullptr));
    CHECK_CU(cuCtxPopCurrent(nullptr));
}

void test_threads(GreenCtxPair& p, volatile int* flag, volatile int* o0, volatile int* o1)
{
    printf("\n--- Test 3: separate OS threads ---\n");
    reset(flag, o0, o1);

    std::thread t0([&]{
        CHECK_CU(cuCtxPushCurrent(p.ctx[0]));
        cudaStream_t s; CHECK_RT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        persistent_kernel<<<1,32,0,s>>>(flag, o0);
        cudaStreamSynchronize(s); cudaStreamDestroy(s);
        CHECK_CU(cuCtxPopCurrent(nullptr));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::thread t1([&]{
        CHECK_CU(cuCtxPushCurrent(p.ctx[1]));
        cudaStream_t s; CHECK_RT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        short_kernel<<<1,32,0,s>>>(flag, o1);
        cudaStreamSynchronize(s); cudaStreamDestroy(s);
        CHECK_CU(cuCtxPopCurrent(nullptr));
    });

    if (wait_for(o0, 42, 3000)) printf("  PASS\n");
    else { printf("  FAIL (timeout)\n"); *flag=1; }
    t0.join(); t1.join();
}

void test_independent(GreenCtxPair& p, volatile int* flag, volatile int* o0, volatile int* o1)
{
    printf("\n--- Test 4: independent kernel (no shared mem) ---\n");
    reset(flag, o0, o1);

    CHECK_CU(cuCtxPushCurrent(p.ctx[0]));
    cudaStream_t s0; CHECK_RT(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    persistent_kernel<<<1,32,0,s0>>>(flag, o0);
    CHECK_RT(cudaGetLastError());
    CHECK_CU(cuCtxPopCurrent(nullptr));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    CHECK_CU(cuCtxPushCurrent(p.ctx[1]));
    cudaStream_t s1; CHECK_RT(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    independent_kernel<<<1,32,0,s1>>>(o1);
    CHECK_RT(cudaGetLastError());
    CHECK_CU(cuCtxPopCurrent(nullptr));

    if (wait_for(o1, 77, 3000)) printf("  PASS\n");
    else printf("  FAIL (timeout)\n");

    *flag=1;
    CHECK_CU(cuCtxPushCurrent(p.ctx[0])); cudaStreamSynchronize(s0); cudaStreamDestroy(s0); CHECK_CU(cuCtxPopCurrent(nullptr));
    CHECK_CU(cuCtxPushCurrent(p.ctx[1])); cudaStreamSynchronize(s1); cudaStreamDestroy(s1); CHECK_CU(cuCtxPopCurrent(nullptr));
}

int main()
{
    printf("=== Green Context Concurrent Execution Tests ===\n");
    printf("CUDA_DEVICE_MAX_CONNECTIONS = %s\n",
        getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "(default)");

    CUdevice device;
    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&device, 0));
    cudaDeviceProp prop;
    CHECK_RT(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, %d SMs, compute %d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    GreenCtxPair p{};
    if (!create_green_contexts(device, 4, p)) { printf("Failed to create green contexts.\n"); return 1; }
    printf("Green contexts: ctx[0]=%u SMs, ctx[1]=%u SMs\n", p.res[0].sm.smCount, p.res[1].sm.smCount);

    volatile int* flag; volatile int* o0; volatile int* o1;
    CHECK_RT(cudaHostAlloc((void**)&flag, sizeof(int), cudaHostAllocMapped|cudaHostAllocPortable));
    CHECK_RT(cudaHostAlloc((void**)&o0, sizeof(int), cudaHostAllocMapped|cudaHostAllocPortable));
    CHECK_RT(cudaHostAlloc((void**)&o1, sizeof(int), cudaHostAllocMapped|cudaHostAllocPortable));

    test_push_pop(p, flag, o0, o1);
    test_nested(p, flag, o0, o1);
    test_threads(p, flag, o0, o1);
    test_independent(p, flag, o0, o1);

    cudaFreeHost((void*)flag); cudaFreeHost((void*)o0); cudaFreeHost((void*)o1);
    cuGreenCtxDestroy(p.green[0]); cuGreenCtxDestroy(p.green[1]);
    printf("\n=== Done ===\n");
    return 0;
}
