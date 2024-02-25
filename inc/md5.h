#pragma once

#include <inttypes.h>

#ifdef __cplusplus
#define restrict

extern "C" {
#endif

/// Example:
/// ```C
/// uint8_t digest[16]; // also a uint64_t[2];
/// md5(&ze_struct, sizeof(the_struct), digest);
/// ```
void md5(uint8_t *const restrict, uint32_t, uint8_t *restrict);

#ifdef __cplusplus
}
#undef restrict
#endif

