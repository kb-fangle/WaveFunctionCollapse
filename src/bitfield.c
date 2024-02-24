#include "bitfield.h"

/// Copy-Pasta from internet
static inline uint64_t
find_nth_set_bit(uint64_t mask, uint64_t n)
{
    uint64_t t, i = n, r = 0;
    const uint64_t m1  = 0x5555555555555555ULL; // even bits
    const uint64_t m2  = 0x3333333333333333ULL; // even 2-bit groups
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0fULL; // even nibbles
    const uint64_t m8  = 0x00ff00ff00ff00ffULL; // even bytes
    const uint64_t c1  = mask;
    const uint64_t c2  = c1 - ((c1 >> 1) & m1);
    const uint64_t c4  = ((c2 >> 2) & m2) + (c2 & m2);
    const uint64_t c8  = ((c4 >> 4) + c4) & m4;
    const uint64_t c16 = ((c8 >> 8) + c8) & m8;
    const uint64_t c32 = (c16 >> 16) + c16;
    const uint64_t c64 = (uint64_t)(((c32 >> 32) + c32) & 0x7f);
    t                  = (c32)&0x3f;
    if (i >= t) {
        r += 32;
        i -= t;
    }
    t = (c16 >> r) & 0x1f;
    if (i >= t) {
        r += 16;
        i -= t;
    }
    t = (c8 >> r) & 0x0f;
    if (i >= t) {
        r += 8;
        i -= t;
    }
    t = (c4 >> r) & 0x07;
    if (i >= t) {
        r += 4;
        i -= t;
    }
    t = (c2 >> r) & 0x03;
    if (i >= t) {
        r += 2;
        i -= t;
    }
    t = (c1 >> r) & 0x01;
    if (i >= t) {
        r += 1;
    }
    if (n >= c64) {
        abort();
    }
    return r;
}

uint64_t
bitfield_only_nth_set(uint64_t x, uint8_t n)
{
    return 1llu << find_nth_set_bit(x, n);
}

