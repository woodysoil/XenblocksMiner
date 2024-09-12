/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "kernelrunner.h"

#include <stdexcept>

#include "CudaException.h"

#define ARGON2_D  0
#define ARGON2_I  1
#define ARGON2_ID 2

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

using namespace std;

__device__ __forceinline__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ __forceinline__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ __forceinline__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}

__device__ __forceinline__ uint64_t u64_shuffle(uint64_t v, uint32_t thread_src)
{
    uint32_t lo = __shfl_sync(0xFFFFFFFF, (uint32_t)v, thread_src);
    uint32_t hi = __shfl_sync(0xFFFFFFFF, (uint32_t)(v >> 32), thread_src);
    return ((uint64_t)hi << 32) | lo;
}

struct __align__(16) block_g {
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct __align__(16) block_l {
    uint32_t lo[ARGON2_QWORDS_IN_BLOCK];
    uint32_t hi[ARGON2_QWORDS_IN_BLOCK];
};

struct __align__(32) block_th {
    uint64_t a, b, c, d;
};

__device__ __forceinline__ uint64_t cmpeq_mask(uint32_t test, uint32_t ref)
{
    uint32_t x = -(test == ref);
    return ((uint64_t)x << 32) | x;
}

__device__ uint64_t block_th_get(const struct block_th *b, uint32_t idx)
{
    uint64_t res = 0;
    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;
    return res;
}

__device__ void block_th_set(struct block_th *b, uint32_t idx, uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

__device__ void move_block(struct block_th *dst, const struct block_th *src)
{
    *dst = *src;
}

__device__ void xor_block(struct block_th *dst, const struct block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__ void load_block(struct block_th *dst, const struct block_g *src,
                           uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void load_block_xor(struct block_th *dst, const struct block_g *src,
                               uint32_t thread)
{
    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void store_block(struct block_g *dst, const struct block_th *src,
                            uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;
}

__device__ void block_l_store(struct block_l* dst, const struct block_th* src,
    uint32_t thread)
{
    dst->lo[0 * THREADS_PER_LANE + thread] = u64_lo(src->a);
    dst->hi[0 * THREADS_PER_LANE + thread] = u64_hi(src->a);

    dst->lo[1 * THREADS_PER_LANE + thread] = u64_lo(src->b);
    dst->hi[1 * THREADS_PER_LANE + thread] = u64_hi(src->b);

    dst->lo[2 * THREADS_PER_LANE + thread] = u64_lo(src->c);
    dst->hi[2 * THREADS_PER_LANE + thread] = u64_hi(src->c);

    dst->lo[3 * THREADS_PER_LANE + thread] = u64_lo(src->d);
    dst->hi[3 * THREADS_PER_LANE + thread] = u64_hi(src->d);
}

__device__ void block_l_load_xor(struct block_th* dst,
    const struct block_l* src, uint32_t thread)
{
    uint32_t lo, hi;

    lo = src->lo[0 * THREADS_PER_LANE + thread];
    hi = src->hi[0 * THREADS_PER_LANE + thread];
    dst->a ^= u64_build(hi, lo);

    lo = src->lo[1 * THREADS_PER_LANE + thread];
    hi = src->hi[1 * THREADS_PER_LANE + thread];
    dst->b ^= u64_build(hi, lo);

    lo = src->lo[2 * THREADS_PER_LANE + thread];
    hi = src->hi[2 * THREADS_PER_LANE + thread];
    dst->c ^= u64_build(hi, lo);

    lo = src->lo[3 * THREADS_PER_LANE + thread];
    hi = src->hi[3 * THREADS_PER_LANE + thread];
    dst->d ^= u64_build(hi, lo);
}
__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = u64_lo(x);
    uint32_t ylo = u64_lo(y);
    return x + y + 2 * u64_build(__umulhi(xlo, ylo), xlo * ylo);
}

__device__ void g1(struct block_th *block)
{
    uint64_t a, b, c, d;
    a = block->a;
    b = block->b;
    c = block->c;
    d = block->d;

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->a = a;
    block->b = b;
    block->c = c;
    block->d = d;
}
__device__
void g(block_th* block)
{
    asm("{"
        ".reg .u64 s, x;"
        ".reg .u32 l1, l2, h1, h2;"
        // a = f(a, b);
        "add.u64 s, %0, %1;"            // s = a + b
        "cvt.u32.u64 l1, %0;"           // xlo = u64_lo(a)
        "cvt.u32.u64 l2, %1;"           // ylo = u64_lo(b)
        "mul.hi.u32 h1, l1, l2;"        // umulhi(xlo, ylo)
        "mul.lo.u32 l1, l1, l2;"        // xlo * ylo
        "mov.b64 x, {l1, h1};"          // x = u64_build(umulhi(xlo, ylo), xlo * ylo)
        "shl.b64 x, x, 1;"              // x = 2 * x
        "add.u64 %0, s, x;"             // a = s + x
        // d = rotr64(d ^ a, 32);
        "xor.b64 x, %3, %0;"
        "mov.b64 {h2, l2}, x;"
        "mov.b64 %3, {l2, h2};"         // swap hi and lo = rotr64(x, 32)
        // c = f(c, d);
        "add.u64 s, %2, %3;"
        "cvt.u32.u64 l1, %2;"
        "mul.hi.u32 h1, l1, l2;"
        "mul.lo.u32 l1, l1, l2;"
        "mov.b64 x, {l1, h1};"
        "shl.b64 x, x, 1;"
        "add.u64 %2, s, x;"
        // b = rotr64(b ^ c, 24);
        "xor.b64 x, %1, %2;"
        "mov.b64 {l1, h1}, x;"
        "prmt.b32 l2, l1, h1, 0x6543;"  // permute bytes 76543210 => 21076543
        "prmt.b32 h2, l1, h1, 0x2107;"  // rotr64(x, 24)
        "mov.b64 %1, {l2, h2};"
        // a = f(a, b);
        "add.u64 s, %0, %1;"
        "cvt.u32.u64 l1, %0;"
        "mul.hi.u32 h1, l1, l2;"
        "mul.lo.u32 l1, l1, l2;"
        "mov.b64 x, {l1, h1};"
        "shl.b64 x, x, 1;"
        "add.u64 %0, s, x;"
        // d = rotr64(d ^ a, 16);
        "xor.b64 x, %3, %0;"
        "mov.b64 {l1, h1}, x;"
        "prmt.b32 l2, l1, h1, 0x5432;"  // permute bytes 76543210 => 10765432
        "prmt.b32 h2, l1, h1, 0x1076;"  // rotr64(x, 16)
        "mov.b64 %3, {l2, h2};"
        // c = f(c, d);
        "add.u64 s, %2, %3;"
        "cvt.u32.u64 l1, %2;"
        "mul.hi.u32 h1, l1, l2;"
        "mul.lo.u32 l1, l1, l2;"
        "mov.b64 x, {l1, h1};"
        "shl.b64 x, x, 1;"
        "add.u64 %2, s, x;"
        // b = rotr64(b ^ c, 63);
        "xor.b64 x, %1, %2;"
        "shl.b64 s, x, 1;"              // x << 1
        "shr.b64 x, x, 63;"             // x >> 63
        "add.u64 %1, s, x;"             // emits less instructions than "or"
        "}"
        : "+l"(block->a), "+l"(block->b), "+l"(block->c), "+l"(block->d)
    );
}


__device__ void transpose1(struct block_th *block, uint32_t thread)
{
    uint32_t thread_group = (thread & 0x0C) >> 2;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t thr = (i << 2) ^ thread;
        uint32_t idx = thread_group ^ i;

        uint64_t v = block_th_get(block, idx);
        v = u64_shuffle(v, thr);
        block_th_set(block, idx, v);
    }
}

__device__ void transpose(
    block_th* block,
    const uint32_t thread)
{
    // thread groups, previously: thread_group = (thread & 0x0C) >> 2
    const uint32_t g1 = (thread & 0x4);
    const uint32_t g2 = (thread & 0x8);

    uint64_t x1 = (g2 ? (g1 ? block->c : block->d) : (g1 ? block->a : block->b));
    uint64_t x2 = (g2 ? (g1 ? block->b : block->a) : (g1 ? block->d : block->c));
    uint64_t x3 = (g2 ? (g1 ? block->a : block->b) : (g1 ? block->c : block->d));

#if CUDART_VERSION < 9000
    x1 = __shfl_xor(x1, 0x4);
    x2 = __shfl_xor(x2, 0x8);
    x3 = __shfl_xor(x3, 0xC);
#else
    x1 = __shfl_xor_sync(0xFFFFFFFF, x1, 0x4);
    x2 = __shfl_xor_sync(0xFFFFFFFF, x2, 0x8);
    x3 = __shfl_xor_sync(0xFFFFFFFF, x3, 0xC);
#endif

    block->a = (g2 ? (g1 ? x3 : x2) : (g1 ? x1 : block->a));
    block->b = (g2 ? (g1 ? x2 : x3) : (g1 ? block->b : x1));
    block->c = (g2 ? (g1 ? x1 : block->c) : (g1 ? x3 : x2));
    block->d = (g2 ? (g1 ? block->d : x1) : (g1 ? x2 : x3));
}


__device__
void shift1_shuffle(
    block_th* block,
    const uint32_t thread)
{
    const uint32_t src_thr_b = (thread & 0x1c) | ((thread + 1) & 0x3);
    const uint32_t src_thr_d = (thread & 0x1c) | ((thread + 3) & 0x3);

#if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x2);
    block->d = __shfl(block->d, src_thr_d);
#else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
#endif
}

__device__
void unshift1_shuffle(
    block_th* block,
    const uint32_t thread)
{
    const uint32_t src_thr_b = (thread & 0x1c) | ((thread + 3) & 0x3);
    const uint32_t src_thr_d = (thread & 0x1c) | ((thread + 1) & 0x3);

#if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x2);
    block->d = __shfl(block->d, src_thr_d);
#else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
#endif
}

__device__
void shift2_shuffle(
    block_th* block,
    const uint32_t thread)
{
    const uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    const uint32_t src_thr_b = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);
    const uint32_t src_thr_d = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);

#if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x10);
    block->d = __shfl(block->d, src_thr_d);
#else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
#endif
}

__device__
void unshift2_shuffle(
    block_th* block,
    const uint32_t thread)
{
    const uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    const uint32_t src_thr_b = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);
    const uint32_t src_thr_d = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);

#if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x10);
    block->d = __shfl(block->d, src_thr_d);
#else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
#endif
}

__device__ void shuffle_block(
    block_th* block,
    const uint32_t thread)
{
    transpose(block, thread);

    g(block);

    shift1_shuffle(block, thread);

    g(block);

    unshift1_shuffle(block, thread);
    transpose(block, thread);

    g(block);

    shift2_shuffle(block, thread);

    g(block);

    unshift2_shuffle(block, thread);
}
__device__ void next_addresses(struct block_th *addr, struct block_th*tmp,
                               uint32_t thread_input, uint32_t thread)
{
    addr->a = u64_build(0, thread_input);
    addr->b = 0;
    addr->c = 0;
    addr->d = 0;

    shuffle_block(addr, thread);

    addr->a ^= u64_build(0, thread_input);
    move_block(tmp, addr);

    shuffle_block(addr, thread);

    xor_block(addr, tmp);
}

__device__ void next_addresses1(struct block_th* addr, struct block_l* tmp,
    uint32_t thread_input, uint32_t thread)
{
    addr->a = u64_build(0, thread_input);
    addr->b = 0;
    addr->c = 0;
    addr->d = 0;

    shuffle_block(addr, thread);

    addr->a ^= u64_build(0, thread_input);
    block_l_store(tmp, addr, thread);

    shuffle_block(addr, thread);

    block_l_load_xor(addr, tmp, thread);
}

__device__ __forceinline__ void compute_ref_pos(
        uint32_t segment_blocks,
        uint32_t slice, uint32_t offset,
        uint32_t *ref_index)
{
    uint32_t ref_area_size = slice * segment_blocks + offset - 1;

    uint32_t index = *ref_index;
    index = __umulhi(index, index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, index);
}


__device__ void argon2_core(
    struct block_g* memory, struct block_g* mem_curr,
    struct block_th* prev, struct block_l* tmp,
    uint32_t thread, 
    uint32_t ref_index)
{
    struct block_g* mem_ref = memory + ref_index;

    load_block_xor(prev, mem_ref, thread);
    block_l_store(tmp, prev, thread);

    shuffle_block(prev, thread);

    block_l_load_xor(prev, tmp, thread);

    store_block(mem_curr, prev, thread);
}

__device__ void argon2_step1(
    struct block_g* memory, struct block_g* mem_curr,
    struct block_th* prev, struct block_l* tmp, struct block_th* addr,
    uint32_t segment_blocks, uint32_t thread,
    uint32_t* thread_input, uint32_t slice,
    uint32_t offset)
{
    uint32_t ref_index;

    if (slice < ARGON2_SYNC_POINTS / 2) {
        uint32_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
        if (addr_index == 0) {
            if (thread == 6) {
                ++* thread_input;
            }
            next_addresses1(addr, tmp, *thread_input, thread);
        }

        uint32_t thr = addr_index % THREADS_PER_LANE;
        uint32_t idx = addr_index / THREADS_PER_LANE;

        uint64_t v = block_th_get(addr, idx);
        v = u64_shuffle(v, thr);
        ref_index = u64_lo(v);
    }
    else {
        uint64_t v = u64_shuffle(prev->a, 0);
        ref_index = u64_lo(v);
    }

    compute_ref_pos(segment_blocks, slice, offset, &ref_index);

    argon2_core(memory, mem_curr, prev, tmp, thread, ref_index);
}

__global__ void argon2_kernel_oneshot(
        struct block_g * __restrict__ memory,
        uint32_t segment_blocks)
{
    extern __shared__ struct block_l shared;

    uint32_t job_id = blockIdx.x;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    memory += (size_t)job_id * lane_blocks;

    struct block_th prev, addr;
    struct block_l* tmp = &shared;
    uint32_t thread_input;

    thread_input = (thread == 3) * lane_blocks + (thread == 4) + (thread == 5) * 2 + (thread == 6);

    next_addresses1(&addr, tmp, thread_input, thread);

    struct block_g *mem_lane = memory;
    struct block_g *mem_prev = mem_lane + 1;
    struct block_g *mem_curr = mem_lane + 2;

    load_block(&prev, mem_prev, thread);

    uint32_t skip = 2;
    //#pragma unroll 4
    for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
        for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
            if (skip > 0) {
                --skip;
                continue;
            }

            argon2_step1(
                        memory, mem_curr, &prev, tmp, &addr,
                        segment_blocks, thread, &thread_input,
                        slice, offset);

            mem_curr ++;
        }

        if (thread == 2) {
            ++thread_input;
        }
        if (thread == 6) {
            thread_input = 0;
        }
    }
    mem_curr = mem_lane;
}

KernelRunner::KernelRunner(uint32_t type, uint32_t version, uint32_t passes,
                           uint32_t lanes, uint32_t segmentBlocks,
                           size_t batchSize)
    : type(type), version(version), passes(passes), lanes(lanes),
          segmentBlocks(segmentBlocks), batchSize(batchSize), stream(), memory(), refs(),
          start(), end(), kernelStart(), kernelEnd(),
          blocksIn(nullptr), blocksOut(nullptr)
{

}

void KernelRunner::init(std::size_t batchSize_){
    batchSize = batchSize_;
    blocksIn = std::make_unique<uint8_t[]>(batchSize * 1 * 2 * ARGON2_BLOCK_SIZE);
    blocksOut = std::make_unique<uint8_t[]>(batchSize * 1 * ARGON2_BLOCK_SIZE);
    size_t memorySize = batchSize * 1 * segmentBlocks * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;

    CudaException::check(cudaMalloc(&memory, memorySize));

    CudaException::check(cudaEventCreate(&start));
    CudaException::check(cudaEventCreate(&end));
    CudaException::check(cudaEventCreate(&kernelStart));
    CudaException::check(cudaEventCreate(&kernelEnd));

    CudaException::check(cudaStreamCreate(&stream));
}

KernelRunner::~KernelRunner()
{
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (end != nullptr) {
        cudaEventDestroy(end);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (memory != nullptr) {
        cudaFree(memory);
    }
    if (refs != nullptr) {
        cudaFree(refs);
    }
}

void *KernelRunner::getInputMemory(size_t jobId) const
{
    size_t copySize = 1 * 2 * ARGON2_BLOCK_SIZE;
    return blocksIn.get() + jobId * copySize;
}
const void *KernelRunner::getOutputMemory(size_t jobId) const
{
    size_t copySize = 1 * ARGON2_BLOCK_SIZE;
    return blocksOut.get() + jobId * copySize;
}

void KernelRunner::copyInputBlocks()
{
    size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    size_t copySize = 1 * 2 * ARGON2_BLOCK_SIZE;

    CudaException::check(cudaMemcpy2DAsync(
                             memory, jobSize,
                             blocksIn.get(), copySize,
                             copySize, batchSize, cudaMemcpyHostToDevice,
                             stream));
}

void KernelRunner::copyOutputBlocks()
{
    size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    size_t copySize = lanes * ARGON2_BLOCK_SIZE;
    uint8_t *mem = static_cast<uint8_t *>(memory);

    CudaException::check(cudaMemcpy2DAsync(
                             blocksOut.get(), copySize,
                             mem + (jobSize - copySize), jobSize,
                             copySize, batchSize, cudaMemcpyDeviceToHost,
                             stream));
}


void KernelRunner::runKernelOneshot()
{
    struct block_g *memory_blocks = (struct block_g *)memory;
    uint32_t sharedSize = sizeof(struct block_l);
    argon2_kernel_oneshot
            <<<dim3(batchSize), dim3(THREADS_PER_LANE), sharedSize, stream>>>(
                memory_blocks, segmentBlocks);
}

void KernelRunner::run()
{
    CudaException::check(cudaEventRecord(start, stream));

    copyInputBlocks();

    CudaException::check(cudaEventRecord(kernelStart, stream));
    
    runKernelOneshot();

    CudaException::check(cudaGetLastError());

    CudaException::check(cudaEventRecord(kernelEnd, stream));

    copyOutputBlocks();

    CudaException::check(cudaEventRecord(end, stream));
}

float KernelRunner::finish()
{
    float time = 0.0;
    CudaException::check(cudaStreamSynchronize(stream));
    CudaException::check(cudaEventElapsedTime(&time, kernelStart, kernelEnd));
    return time;
}

