//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include "Function_Dropout.h"

#include "../LSDN_CudaCommon.h"

#include "cuda_runtime.h"
#include "curand.h"

template <typename T>
__global__ void DropoutKernel(const int n, const T* in, const unsigned int* mask, const unsigned int threshold, const T scale, T* out) {
	CUDA_KERNEL_LOOP(index, n) {
		out[index] = in[index] * (mask[index] > threshold) * scale;
	}
}

template <typename T>
__global__ void DropoutAddKernel(const int n, const T* in, const unsigned int* mask, const unsigned int threshold, const T scale, T* out) {
	CUDA_KERNEL_LOOP(index, n) {
		out[index] += in[index] * (mask[index] > threshold) * scale;
	}
}

template <class N>
void DropoutFunction<N>::MaskOperation(i2t<true>, ValueType* input, ValueType* output, unsigned int threshold, SizeType Length, bool addToOutput) {
	if (addToOutput) {
		DropoutAddKernel<ValueType><<<LSDN_GET_BLOCKS(Length), LSDN_CUDA_NUM_THREADS>>>(Length, input, mask_, threshold, scale_, output);
	} else {
		DropoutKernel<ValueType><<<LSDN_GET_BLOCKS(Length), LSDN_CUDA_NUM_THREADS>>>(Length, input, mask_, threshold, scale_, output);
	}
	check_cuda_errors(__FILE__, __LINE__);
}

template class DropoutFunction<Node<double, int, false> >;
template class DropoutFunction<Node<double, int, true> >;
template class DropoutFunction<Node<float, int, false> >;
template class DropoutFunction<Node<float, int, true> >;