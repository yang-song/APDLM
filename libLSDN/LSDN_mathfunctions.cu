//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include "LSDN_mathfunctions.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"

template <class T>
__global__ void set_kernel(const size_t n, T* dst, const T val) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = val;
	}
}
template __global__ void set_kernel<double>(const size_t, double*, const double);
template __global__ void set_kernel<float>(const size_t, float*, const float);

template <class T>
void LSDNMemSet(i2t<true>, T* dst, T val, size_t numEl) {
	if (val == 0) {
		cudaMemset((void*)dst, 0, sizeof(T)*numEl);
	} else {
		set_kernel<T><<<LSDN_GET_BLOCKS(numEl), LSDN_CUDA_NUM_THREADS>>>(numEl, dst, val);
	}
	check_cuda_errors(__FILE__, __LINE__);
}
template void LSDNMemSet<double>(i2t<true>, double*, double, size_t);
template void LSDNMemSet<float>(i2t<true>, float*, float, size_t);

template <typename T>
__global__ void kernel_exp(const size_t num, const T* data_in, T* data_out) {
	CUDA_KERNEL_LOOP(index, num) {
		data_out[index] = exp(data_in[index]);
	}
}
template __global__ void kernel_exp<double>(const size_t, const double*, double*);
template __global__ void kernel_exp<float>(const size_t, const float*, float*);

template <class T>
void ElementwiseExp(i2t<true>, T* data_in, T* data_out, size_t numEl) {
	kernel_exp<T><<<LSDN_GET_BLOCKS(numEl), LSDN_CUDA_NUM_THREADS>>>(numEl, data_in, data_out);
	check_cuda_errors(__FILE__, __LINE__);
}
template void ElementwiseExp<double>(i2t<true>, double*, double*, size_t);
template void ElementwiseExp<float>(i2t<true>, float*, float*, size_t);

void LSDNMemCpy(i2t<true>, void* dst, void* src, size_t bytes) {
	cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_NormLimitByCol(int num, int row, T norm_limit, T* in) {
	CUDA_KERNEL_LOOP(index, num) {
		T s = 0;

		for (int r = 0; r < row; ++r) {
			s += in[r + index*row] * in[r + index*row];
		}
		s = sqrt(s);

		T scale = (s > norm_limit) ? norm_limit / s : 1;

		if (scale != 1) {
			for (int r = 0; r < row; ++r) {
				in[r + index*row] *= scale;
			}
		}
	}
}

template <class T>
void NormLimitByCol(i2t<true>, int row, int col, T norm_limit, T* in)
{
	kernel_NormLimitByCol<T><<<LSDN_GET_BLOCKS(col), LSDN_CUDA_NUM_THREADS>>>(col, row, norm_limit, in);
	check_cuda_errors(__FILE__, __LINE__);
}

template void NormLimitByCol<double>(i2t<true>, int, int, double, double*);
template void NormLimitByCol<float>(i2t<true>, int, int, float, float*);