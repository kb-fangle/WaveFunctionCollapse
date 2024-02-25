#pragma once

#include <stdint.h>
#include <stdio.h>

#ifdef __CUDACC__
#define WFC_DEVICE __device__
#define WFC_HOST __host__
#else
#define WFC_DEVICE
#define WFC_HOST
#endif

/// Set the specific bit in an integer.
static inline WFC_DEVICE WFC_HOST uint64_t
bitfield_set(uint64_t flag, uint8_t index)
{
    return flag | (1llu << index);
}

/// Unset the specific bit in an integer.
static inline WFC_DEVICE WFC_HOST uint64_t
bitfield_unset(uint64_t flag, uint8_t index)
{
    return flag & ~(1llu << index);
}

/// Get the specific bit in an integer. If the index doesn't exists returns 0.
static inline WFC_DEVICE WFC_HOST uint64_t
bitfield_get(uint64_t flag, uint8_t index)
{
    return (flag >> index) & 1llu;
}

/// Count the number of set bits in an integer.
#if defined(__has_builtin)
#if __has_builtin(__builtin_popcountll)
#define bitfield_count(x) ((uint8_t)__builtin_popcountll((x)))
#endif
#endif

#ifndef bitfield_count
static inline WFC_DEVICE WFC_HOST uint8_t
bitfield_count_(uint64_t x)
{
    const uint64_t m1 = 0x5555555555555555;  // binary: 0101...
    const uint64_t m2 = 0x3333333333333333;  // binary: 00110011..
    const uint64_t m4 = 0x0f0f0f0f0f0f0f0f;  // binary:  4 zeros,  4 ones ...
    const uint64_t h01 = 0x0101010101010101; // the sum of 256 to the power of
                                             // 0,1,2,3...

    x -= (x >> 1) & m1;             // put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2); // put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & m4;        // put count of each 8 bits into those 8 bits
    // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
    return (uint8_t)((x * h01) >> 56);
}

#define bitfield_count(x) bitfield_count_((x))
#endif

/// Get the integer with only the nth setted bit of the said integer.
uint64_t bitfield_only_nth_set(uint64_t, uint8_t);

/// Get the position of the nth set bit in a 64 bit number
static inline WFC_DEVICE WFC_HOST uint32_t
bitfield_get_nth_set_bit(uint64_t v, uint32_t r)
{
    uint32_t i = 0;
    while (r > 0) {
        i++;
        r -= v & 1;
        v >>= 1;
    }
    return i;
}

/// Prints the bitfield to the file descriptor.
void bitfield_print(FILE *const, uint64_t);

#undef WFC_HOST
#undef WFC_DEVICE
