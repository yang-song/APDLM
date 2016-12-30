//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __LSDN_CUDACOMMON_H__
#define __LSDN_CUDACOMMON_H__
#include <cstdio>
#include <assert.h>
#include <cublas_v2.h>

#if __CUDA_ARCH__ >= 200
#define LSDN_CUDA_NUM_THREADS 1024
#else
#define LSDN_CUDA_NUM_THREADS 512
#endif

// CUDA: number of blocks for threads.
#define LSDN_GET_BLOCKS(N) ((N)+LSDN_CUDA_NUM_THREADS-1)/LSDN_CUDA_NUM_THREADS

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	i < (n); \
	i += blockDim.x * gridDim.x)

#ifdef _DEBUG
inline void check_cuda_errors(const char *filename, const int line_number) {
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
#ifdef _MSC_VER
		__debugbreak();
#endif
		//while(true) {}
		assert(false);
		exit(-1);
	}
}
#else
inline void check_cuda_errors(const char*, const int) {}
#endif

#ifdef _DEBUG
static const char *cublas_GetErrorEnum(cublasStatus_t error)
{
	switch (error) {
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
	return "<unknown>";
}

inline void check_cublas_errors(const char *filename, const int line_number, cublasStatus_t error) {
	if (error != CUBLAS_STATUS_SUCCESS) {
		printf("CUDA error at %s:%i: %s\n", filename, line_number, cublas_GetErrorEnum(error));
		check_cuda_errors(filename, line_number);
		assert(false);
		exit(-1);
	}
	check_cuda_errors(filename, line_number);
}
#else
inline void check_cublas_errors(const char*, const int, cublasStatus_t) {}
#endif

#endif
