/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "kernelrunner.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "CudaException.h"

#define ARGON2_D  0
#define ARGON2_I  1
#define ARGON2_ID 2

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4
#define ARGON2_PREHASH_DIGEST_LENGTH 64
#define ARGON2_PREHASH_SEED_LENGTH 72
#define BLAKE2B_BLOCK_BYTES 128
#define BLAKE2B_OUT_BYTES 64

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

__device__ __forceinline__ void device_store32(void* dst, uint32_t v)
{
    uint8_t* out = static_cast<uint8_t*>(dst);
    out[0] = static_cast<uint8_t>(v);
    out[1] = static_cast<uint8_t>(v >> 8);
    out[2] = static_cast<uint8_t>(v >> 16);
    out[3] = static_cast<uint8_t>(v >> 24);
}

__device__ __forceinline__ void device_store64(void* dst, uint64_t v)
{
    uint8_t* out = static_cast<uint8_t*>(dst);
    out[0] = static_cast<uint8_t>(v);
    out[1] = static_cast<uint8_t>(v >> 8);
    out[2] = static_cast<uint8_t>(v >> 16);
    out[3] = static_cast<uint8_t>(v >> 24);
    out[4] = static_cast<uint8_t>(v >> 32);
    out[5] = static_cast<uint8_t>(v >> 40);
    out[6] = static_cast<uint8_t>(v >> 48);
    out[7] = static_cast<uint8_t>(v >> 56);
}

__device__ __forceinline__ uint64_t device_load64(const void* src)
{
    const uint8_t* in = static_cast<const uint8_t*>(src);
    return static_cast<uint64_t>(in[0]) |
           (static_cast<uint64_t>(in[1]) << 8) |
           (static_cast<uint64_t>(in[2]) << 16) |
           (static_cast<uint64_t>(in[3]) << 24) |
           (static_cast<uint64_t>(in[4]) << 32) |
           (static_cast<uint64_t>(in[5]) << 40) |
           (static_cast<uint64_t>(in[6]) << 48) |
           (static_cast<uint64_t>(in[7]) << 56);
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

struct Blake2bDeviceState {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t buf[BLAKE2B_BLOCK_BYTES];
    uint32_t buf_len;
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

__device__ __forceinline__ uint64_t block_th_get_uniform(const struct block_th *b, uint32_t idx)
{
    switch (idx) {
    case 0:
        return b->a;
    case 1:
        return b->b;
    case 2:
        return b->c;
    default:
        return b->d;
    }
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

__device__ __constant__ uint64_t blake2b_iv_device[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

__device__ __constant__ uint8_t blake2b_sigma_device[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

__device__ __forceinline__ uint64_t device_blake_rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ void device_blake_g(
    const uint64_t* m, uint32_t r, uint32_t i,
    uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d)
{
    a = a + b + m[blake2b_sigma_device[r][2 * i + 0]];
    d = device_blake_rotr64(d ^ a, 32);
    c = c + d;
    b = device_blake_rotr64(b ^ c, 24);
    a = a + b + m[blake2b_sigma_device[r][2 * i + 1]];
    d = device_blake_rotr64(d ^ a, 16);
    c = c + d;
    b = device_blake_rotr64(b ^ c, 63);
}

__device__ __forceinline__ void device_blake_round(uint64_t* m, uint64_t* v, uint32_t r)
{
    device_blake_g(m, r, 0, v[0], v[4], v[8], v[12]);
    device_blake_g(m, r, 1, v[1], v[5], v[9], v[13]);
    device_blake_g(m, r, 2, v[2], v[6], v[10], v[14]);
    device_blake_g(m, r, 3, v[3], v[7], v[11], v[15]);
    device_blake_g(m, r, 4, v[0], v[5], v[10], v[15]);
    device_blake_g(m, r, 5, v[1], v[6], v[11], v[12]);
    device_blake_g(m, r, 6, v[2], v[7], v[8], v[13]);
    device_blake_g(m, r, 7, v[3], v[4], v[9], v[14]);
}

__device__ void device_blake2b_compress(Blake2bDeviceState* state, const void* block, uint64_t f0)
{
    uint64_t m[16];
    uint64_t v[16];
    const uint8_t* in = static_cast<const uint8_t*>(block);

#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        m[i] = device_load64(in + i * sizeof(uint64_t));
    }

#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        v[i] = state->h[i];
        v[i + 8] = blake2b_iv_device[i];
    }
    v[12] ^= state->t[0];
    v[13] ^= state->t[1];
    v[14] ^= f0;

#pragma unroll
    for (uint32_t r = 0; r < 12; ++r) {
        device_blake_round(m, v, r);
    }

#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        state->h[i] ^= v[i] ^ v[i + 8];
    }
}

__device__ void device_blake2b_increment_counter(Blake2bDeviceState* state, uint64_t inc)
{
    state->t[0] += inc;
    state->t[1] += (state->t[0] < inc);
}

__device__ void device_blake2b_init(Blake2bDeviceState* state, uint32_t out_len)
{
    state->t[0] = 0;
    state->t[1] = 0;
    state->buf_len = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        state->h[i] = blake2b_iv_device[i];
    }
    state->h[0] ^= static_cast<uint64_t>(out_len) |
                   (UINT64_C(1) << 16) | (UINT64_C(1) << 24);
}

__device__ void device_blake2b_update(Blake2bDeviceState* state, const void* input, uint32_t input_len)
{
    const uint8_t* in = static_cast<const uint8_t*>(input);
    if (state->buf_len + input_len > BLAKE2B_BLOCK_BYTES) {
        const uint32_t have = state->buf_len;
        const uint32_t left = BLAKE2B_BLOCK_BYTES - have;
        for (uint32_t i = 0; i < left; ++i) {
            state->buf[have + i] = in[i];
        }

        device_blake2b_increment_counter(state, BLAKE2B_BLOCK_BYTES);
        device_blake2b_compress(state, state->buf, 0);

        state->buf_len = 0;
        input_len -= left;
        in += left;

        while (input_len > BLAKE2B_BLOCK_BYTES) {
            device_blake2b_increment_counter(state, BLAKE2B_BLOCK_BYTES);
            device_blake2b_compress(state, in, 0);
            input_len -= BLAKE2B_BLOCK_BYTES;
            in += BLAKE2B_BLOCK_BYTES;
        }
    }
    for (uint32_t i = 0; i < input_len; ++i) {
        state->buf[state->buf_len + i] = in[i];
    }
    state->buf_len += input_len;
}

__device__ void device_blake2b_final(Blake2bDeviceState* state, void* out, uint32_t out_len)
{
    device_blake2b_increment_counter(state, state->buf_len);
    for (uint32_t i = state->buf_len; i < BLAKE2B_BLOCK_BYTES; ++i) {
        state->buf[i] = 0;
    }
    device_blake2b_compress(state, state->buf, UINT64_C(0xFFFFFFFFFFFFFFFF));

    uint8_t buffer[BLAKE2B_OUT_BYTES];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        device_store64(buffer + i * sizeof(uint64_t), state->h[i]);
    }
    uint8_t* output = static_cast<uint8_t*>(out);
    for (uint32_t i = 0; i < out_len; ++i) {
        output[i] = buffer[i];
    }
}

__device__ void device_digest_long(void* out, uint32_t out_len, const void* input, uint32_t input_len)
{
    uint8_t* output = static_cast<uint8_t*>(out);
    uint8_t out_len_bytes[sizeof(uint32_t)];
    uint8_t out_buffer[BLAKE2B_OUT_BYTES];
    Blake2bDeviceState blake;

    device_store32(out_len_bytes, out_len);
    if (out_len <= BLAKE2B_OUT_BYTES) {
        device_blake2b_init(&blake, out_len);
        device_blake2b_update(&blake, out_len_bytes, sizeof(out_len_bytes));
        device_blake2b_update(&blake, input, input_len);
        device_blake2b_final(&blake, out, out_len);
        return;
    }

    device_blake2b_init(&blake, BLAKE2B_OUT_BYTES);
    device_blake2b_update(&blake, out_len_bytes, sizeof(out_len_bytes));
    device_blake2b_update(&blake, input, input_len);
    device_blake2b_final(&blake, out_buffer, BLAKE2B_OUT_BYTES);

    for (uint32_t i = 0; i < BLAKE2B_OUT_BYTES / 2; ++i) {
        *output++ = out_buffer[i];
    }

    uint32_t to_produce = out_len - BLAKE2B_OUT_BYTES / 2;
    while (to_produce > BLAKE2B_OUT_BYTES) {
        device_blake2b_init(&blake, BLAKE2B_OUT_BYTES);
        device_blake2b_update(&blake, out_buffer, BLAKE2B_OUT_BYTES);
        device_blake2b_final(&blake, out_buffer, BLAKE2B_OUT_BYTES);

        for (uint32_t i = 0; i < BLAKE2B_OUT_BYTES / 2; ++i) {
            *output++ = out_buffer[i];
        }
        to_produce -= BLAKE2B_OUT_BYTES / 2;
    }

    device_blake2b_init(&blake, to_produce);
    device_blake2b_update(&blake, out_buffer, BLAKE2B_OUT_BYTES);
    device_blake2b_final(&blake, output, to_produce);
}

__device__ void device_initial_hash(void* out,
                                    const uint8_t* password,
                                    uint32_t password_len,
                                    const uint8_t* salt,
                                    uint32_t salt_len,
                                    uint32_t output_len,
                                    uint32_t memory_cost,
                                    uint32_t time_cost,
                                    uint32_t version,
                                    uint32_t type,
                                    uint32_t lanes)
{
    Blake2bDeviceState blake;
    device_blake2b_init(&blake, ARGON2_PREHASH_DIGEST_LENGTH);

    uint8_t header[7 * sizeof(uint32_t)];
    device_store32(header + 0 * sizeof(uint32_t), lanes);
    device_store32(header + 1 * sizeof(uint32_t), output_len);
    device_store32(header + 2 * sizeof(uint32_t), memory_cost);
    device_store32(header + 3 * sizeof(uint32_t), time_cost);
    device_store32(header + 4 * sizeof(uint32_t), version);
    device_store32(header + 5 * sizeof(uint32_t), type);
    device_store32(header + 6 * sizeof(uint32_t), password_len);
    device_blake2b_update(&blake, header, sizeof(header));
    device_blake2b_update(&blake, password, password_len);

    uint8_t value[sizeof(uint32_t)];
    device_store32(value, salt_len);
    device_blake2b_update(&blake, value, sizeof(value));
    device_blake2b_update(&blake, salt, salt_len);

    device_store32(value, 0);
    device_blake2b_update(&blake, value, sizeof(value));
    device_blake2b_update(&blake, value, sizeof(value));

    device_blake2b_final(&blake, out, ARGON2_PREHASH_DIGEST_LENGTH);
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

__device__ void argon2_step_indexed(
    struct block_g* memory, struct block_g* mem_curr,
    struct block_th* prev, struct block_l* tmp, struct block_th* addr,
    uint32_t segment_blocks, uint32_t thread,
    uint32_t* thread_input, uint32_t slice,
    uint32_t offset)
{
    uint32_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
    if (addr_index == 0) {
        if (thread == 6) {
            ++* thread_input;
        }
        next_addresses1(addr, tmp, *thread_input, thread);
    }

    uint32_t thr = addr_index % THREADS_PER_LANE;
    uint32_t idx = addr_index / THREADS_PER_LANE;

    uint64_t v = block_th_get_uniform(addr, idx);
    v = u64_shuffle(v, thr);
    uint32_t ref_index = u64_lo(v);

    compute_ref_pos(segment_blocks, slice, offset, &ref_index);

    argon2_core(memory, mem_curr, prev, tmp, thread, ref_index);
}

__device__ void argon2_step_dependent(
    struct block_g* memory, struct block_g* mem_curr,
    struct block_th* prev, struct block_l* tmp,
    uint32_t segment_blocks, uint32_t thread,
    uint32_t slice, uint32_t offset)
{
    uint64_t v = u64_shuffle(prev->a, 0);
    uint32_t ref_index = u64_lo(v);

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

    for (uint32_t offset = 2; offset < segment_blocks; ++offset) {
        argon2_step_indexed(
                    memory, mem_curr, &prev, tmp, &addr,
                    segment_blocks, thread, &thread_input,
                    0, offset);

        mem_curr ++;
    }

    if (thread == 2) {
        ++thread_input;
    }
    if (thread == 6) {
        thread_input = 0;
    }

    for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
        argon2_step_indexed(
                    memory, mem_curr, &prev, tmp, &addr,
                    segment_blocks, thread, &thread_input,
                    1, offset);

        mem_curr ++;
    }

    for (uint32_t slice = 2; slice < ARGON2_SYNC_POINTS; ++slice) {
        for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
            argon2_step_dependent(
                        memory, mem_curr, &prev, tmp,
                        segment_blocks, thread,
                        slice, offset);

            mem_curr ++;
        }
    }
    mem_curr = mem_lane;
}

__global__ void argon2_first_blocks_kernel(
        struct block_g* __restrict__ memory,
        const uint8_t* __restrict__ keys,
        uint32_t key_length,
        const uint8_t* __restrict__ salt,
        uint32_t salt_length,
        uint32_t output_length,
        uint32_t memory_cost,
        uint32_t time_cost,
        uint32_t version,
        uint32_t type,
        uint32_t lanes,
        uint32_t segment_blocks,
        size_t batch_size)
{
    const size_t job_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (job_id >= batch_size) {
        return;
    }

    const uint8_t* password = keys + job_id * key_length;
    uint8_t init_hash[ARGON2_PREHASH_SEED_LENGTH];
    device_initial_hash(init_hash, password, key_length, salt, salt_length,
                        output_length, memory_cost, time_cost, version, type, lanes);

    const uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;
    uint8_t* output = reinterpret_cast<uint8_t*>(memory + job_id * lane_blocks);

    device_store32(init_hash + ARGON2_PREHASH_DIGEST_LENGTH, 0);
    device_store32(init_hash + ARGON2_PREHASH_DIGEST_LENGTH + 4, 0);
    device_digest_long(output, ARGON2_BLOCK_SIZE, init_hash, sizeof(init_hash));

    device_store32(init_hash + ARGON2_PREHASH_DIGEST_LENGTH, 1);
    device_store32(init_hash + ARGON2_PREHASH_DIGEST_LENGTH + 4, 0);
    device_digest_long(output + ARGON2_BLOCK_SIZE, ARGON2_BLOCK_SIZE, init_hash, sizeof(init_hash));
}

KernelRunner::KernelRunner(uint32_t type, uint32_t version, uint32_t passes,
                           uint32_t lanes, uint32_t segmentBlocks,
                           size_t batchSize)
    : type(type), version(version), passes(passes), lanes(lanes),
          segmentBlocks(segmentBlocks), allocatedSegmentBlocks(segmentBlocks),
          batchSize(batchSize), stream(), memory(), refs(),
          deviceKeys(), deviceSalt(), deviceKeysCapacity(0), deviceSaltCapacity(0),
          deviceFirstBlocksReady(false), deviceFirstBlockKeyLength(0),
          deviceFirstBlockSaltLength(0), deviceFirstBlockOutputLength(0),
          deviceFirstBlockMemoryCost(0), deviceFirstBlockTimeCost(0),
          deviceFirstBlockVersion(0), deviceFirstBlockType(0),
          deviceFirstBlockLanes(0), lastUsedDeviceFirstBlocks(false),
          start(), end(), copyStart(), copyEnd(), firstBlockStart(), firstBlockEnd(), kernelStart(), kernelEnd(),
          blocksIn(nullptr), blocksOut(nullptr)
{

}

void KernelRunner::init(std::size_t batchSize_){
    batchSize = batchSize_;
    allocatedSegmentBlocks = segmentBlocks;
    blocksIn = std::make_unique<uint8_t[]>(batchSize * 1 * 2 * ARGON2_BLOCK_SIZE);
    blocksOut = std::make_unique<uint8_t[]>(batchSize * 1 * ARGON2_BLOCK_SIZE);
    size_t memorySize = batchSize * 1 * allocatedSegmentBlocks * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;

    CudaException::check(cudaMalloc(&memory, memorySize));

    CudaException::check(cudaEventCreate(&start));
    CudaException::check(cudaEventCreate(&end));
    CudaException::check(cudaEventCreate(&copyStart));
    CudaException::check(cudaEventCreate(&copyEnd));
    CudaException::check(cudaEventCreate(&firstBlockStart));
    CudaException::check(cudaEventCreate(&firstBlockEnd));
    CudaException::check(cudaEventCreate(&kernelStart));
    CudaException::check(cudaEventCreate(&kernelEnd));

    CudaException::check(cudaStreamCreate(&stream));
}

bool KernelRunner::canReuse(uint32_t type_, uint32_t version_,
                            uint32_t passes_, uint32_t lanes_,
                            uint32_t segmentBlocks_,
                            std::size_t batchSize_) const
{
    return type == type_ &&
           version == version_ &&
           passes == passes_ &&
           lanes == lanes_ &&
           batchSize == batchSize_ &&
           segmentBlocks_ <= allocatedSegmentBlocks;
}

void KernelRunner::reconfigure(uint32_t type_, uint32_t version_,
                               uint32_t passes_, uint32_t lanes_,
                               uint32_t segmentBlocks_,
                               std::size_t batchSize_)
{
    type = type_;
    version = version_;
    passes = passes_;
    lanes = lanes_;
    segmentBlocks = segmentBlocks_;
    batchSize = batchSize_;
    deviceFirstBlocksReady = false;
    lastUsedDeviceFirstBlocks = false;
}

KernelRunner::~KernelRunner()
{
    if (stream != nullptr) {
        // Variable-shape benchmark loops recreate runners; drain queued work before tearing down CUDA resources.
        cudaStreamSynchronize(stream);
    }
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (end != nullptr) {
        cudaEventDestroy(end);
    }
    if (copyStart != nullptr) {
        cudaEventDestroy(copyStart);
    }
    if (copyEnd != nullptr) {
        cudaEventDestroy(copyEnd);
    }
    if (firstBlockStart != nullptr) {
        cudaEventDestroy(firstBlockStart);
    }
    if (firstBlockEnd != nullptr) {
        cudaEventDestroy(firstBlockEnd);
    }
    if (kernelStart != nullptr) {
        cudaEventDestroy(kernelStart);
    }
    if (kernelEnd != nullptr) {
        cudaEventDestroy(kernelEnd);
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
    if (deviceKeys != nullptr) {
        cudaFree(deviceKeys);
    }
    if (deviceSalt != nullptr) {
        cudaFree(deviceSalt);
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

bool KernelRunner::prepareInputBlocksOnDevice(const std::vector<std::string>& passwords,
                                              const std::vector<std::uint8_t>& saltBytes,
                                              std::uint32_t outputLength,
                                              std::uint32_t memoryCost,
                                              std::uint32_t timeCost,
                                              std::uint32_t version_,
                                              std::uint32_t type_,
                                              std::uint32_t lanes_)
{
    deviceFirstBlocksReady = false;
    if (passwords.empty() || passwords.size() != batchSize ||
        outputLength != BLAKE2B_OUT_BYTES ||
        timeCost != 1 ||
        lanes_ != 1 ||
        saltBytes.empty() ||
        saltBytes.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }

    const std::size_t keyLength = passwords.front().size();
    if (keyLength == 0 || keyLength > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    for (const std::string& password : passwords) {
        if (password.size() != keyLength) {
            return false;
        }
    }

    const std::size_t keyBytes = keyLength * passwords.size();
    std::vector<std::uint8_t> flatKeys(keyBytes);
    std::uint8_t* cursor = flatKeys.data();
    for (const std::string& password : passwords) {
        std::copy(password.begin(), password.end(), cursor);
        cursor += password.size();
    }

    if (deviceKeysCapacity < keyBytes) {
        if (deviceKeys != nullptr) {
            CudaException::check(cudaFree(deviceKeys));
            deviceKeys = nullptr;
        }
        CudaException::check(cudaMalloc(&deviceKeys, keyBytes));
        deviceKeysCapacity = keyBytes;
    }
    if (deviceSaltCapacity < saltBytes.size()) {
        if (deviceSalt != nullptr) {
            CudaException::check(cudaFree(deviceSalt));
            deviceSalt = nullptr;
        }
        CudaException::check(cudaMalloc(&deviceSalt, saltBytes.size()));
        deviceSaltCapacity = saltBytes.size();
    }

    CudaException::check(cudaMemcpy(deviceKeys, flatKeys.data(), keyBytes,
                                    cudaMemcpyHostToDevice));
    CudaException::check(cudaMemcpy(deviceSalt, saltBytes.data(), saltBytes.size(),
                                    cudaMemcpyHostToDevice));

    deviceFirstBlocksReady = true;
    deviceFirstBlockKeyLength = keyLength;
    deviceFirstBlockSaltLength = static_cast<std::uint32_t>(saltBytes.size());
    deviceFirstBlockOutputLength = outputLength;
    deviceFirstBlockMemoryCost = memoryCost;
    deviceFirstBlockTimeCost = timeCost;
    deviceFirstBlockVersion = version_;
    deviceFirstBlockType = type_;
    deviceFirstBlockLanes = lanes_;
    return true;
}

void KernelRunner::runDeviceFirstBlockKernel()
{
    const std::size_t threadsPerBlock = 128;
    const std::size_t grid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    argon2_first_blocks_kernel
            <<<dim3(grid), dim3(threadsPerBlock), 0, stream>>>(
                static_cast<struct block_g*>(memory),
                static_cast<const uint8_t*>(deviceKeys),
                static_cast<std::uint32_t>(deviceFirstBlockKeyLength),
                static_cast<const uint8_t*>(deviceSalt),
                deviceFirstBlockSaltLength,
                deviceFirstBlockOutputLength,
                deviceFirstBlockMemoryCost,
                deviceFirstBlockTimeCost,
                deviceFirstBlockVersion,
                deviceFirstBlockType,
                deviceFirstBlockLanes,
                segmentBlocks,
                batchSize);
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
    CudaException::check(cudaEventRecord(copyStart, stream));

    const bool useDeviceFirstBlocks = deviceFirstBlocksReady;
    lastUsedDeviceFirstBlocks = useDeviceFirstBlocks;
    if (useDeviceFirstBlocks) {
        CudaException::check(cudaEventRecord(copyEnd, stream));
        CudaException::check(cudaEventRecord(firstBlockStart, stream));
        runDeviceFirstBlockKernel();
        CudaException::check(cudaGetLastError());
        CudaException::check(cudaEventRecord(firstBlockEnd, stream));
    } else {
        copyInputBlocks();
        CudaException::check(cudaEventRecord(copyEnd, stream));
        CudaException::check(cudaEventRecord(firstBlockStart, stream));
        CudaException::check(cudaEventRecord(firstBlockEnd, stream));
    }
    deviceFirstBlocksReady = false;

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

float KernelRunner::getLastHostToDeviceMs() const
{
    float time = 0.0f;
    CudaException::check(cudaEventElapsedTime(&time, copyStart, copyEnd));
    return time;
}

float KernelRunner::getLastGpuFirstBlockMs() const
{
    if (!lastUsedDeviceFirstBlocks) {
        return 0.0f;
    }
    float time = 0.0f;
    CudaException::check(cudaEventElapsedTime(&time, firstBlockStart, firstBlockEnd));
    return time;
}

float KernelRunner::getLastDeviceToHostMs() const
{
    float time = 0.0f;
    CudaException::check(cudaEventElapsedTime(&time, kernelEnd, end));
    return time;
}

