/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a, b) (a < b ? a : b)
#endif

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); getLastCudaError(""); }

#define cudaMallocErr(...) checkCudaErrors(cudaMalloc(__VA_ARGS__))
#define cudaMemcpyErr(...) checkCudaErrors(cudaMemcpy(__VA_ARGS__))
#define cudaMemcpyAsyncErr(...) checkCudaErrors(cudaMemcpyAsync(__VA_ARGS__))
#define cudaMemsetErr(...) checkCudaErrors(cudaMemset(__VA_ARGS__))
#define cudaMemsetAsyncErr(...) checkCudaErrors(cudaMemsetAsync(__VA_ARGS__))
#define cudaStreamSynchronizeErr(...) checkCudaErrors(cudaStreamSynchronize(__VA_ARGS__))
#define cudaMallocHostErr(...) checkCudaErrors(cudaMallocHost(__VA_ARGS__))
#define cudaFreeErr(...) checkCudaErrors(cudaFree(__VA_ARGS__))
#define cudaFreeHostErr(...) checkCudaErrors(cudaFreeHost(__VA_ARGS__))
#define cudaDeviceSynchronizeErr(...) checkCudaErrors(cudaDeviceSynchronize(__VA_ARGS__))

#ifndef WFC_CUDA_NO_CHECK
#define cudaMalloc(...) cudaMallocErr(__VA_ARGS__)
#define cudaMemcpy(...) cudaMemcpyErr(__VA_ARGS__)
#define cudaMemcpyAsync(...) cudaMemcpyAsyncErr(__VA_ARGS__)
#define cudaMemset(...) cudaMemsetErr(__VA_ARGS__)
#define cudaMemsetAsync(...) cudaMemsetAsyncErr(__VA_ARGS__)
#define cudaStreamSynchronize(...) cudaStreamSynchronizeErr(__VA_ARGS__)
#define cudaMallocHost(...) cudaMallocHostErr(__VA_ARGS__)
#define cudaFree(...) cudaFreeErr(__VA_ARGS__)
#define cudaFreeHost(...) cudaFreeHostErr(__VA_ARGS__)
#define cudaDeviceSynchronize(...) cudaDeviceSynchronizeErr(__VA_ARGS__)
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, (int)(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60, 64},
      {0x61, 128},
      {0x62, 128},
      {0x70, 64},
      {0x72, 64},
      {0x75, 64},
      {0x80, 64},
      {0x86, 128},
      {0x87, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char *_ConvertSMVer2ArchName(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char *name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1)
  {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
// end of GPU Architecture definitions

// end of CUDA Helper Functions

#endif // COMMON_HELPER_CUDA_H_
