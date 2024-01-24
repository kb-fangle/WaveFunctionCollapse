#pragma once

#include "inttypes.h"

#include <stdio.h>
#include <stdlib.h>

/// Set the specific bit in an integer.
static inline uint64_t
bitfield_set(uint64_t flag, uint8_t index)
{
    return flag | (1llu << index);
}

/// Get the specific bit in an integer. If the index doesn't exists returns 0.
static inline uint64_t
bitfield_get(uint64_t flag, uint8_t index)
{
    return (flag >> index) & 1llu;
}

/// Count the number of set bits in an integer.
static inline uint8_t
bitfield_count(uint64_t x)
{
    const uint64_t m1  = 0x5555555555555555; //binary: 0101...
    const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...

    x -= (x >> 1) & m1;                //put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2);    //put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & m4;           //put count of each 8 bits into those 8 bits
    return (uint8_t)((x * h01) >> 56); //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

/// Get the integer with only the nth setted bit of the said integer.
uint64_t bitfield_only_nth_set(uint64_t, uint8_t);

/// Prints the bitfield to the file descriptor.
void bitfield_print(FILE *const, uint64_t);
