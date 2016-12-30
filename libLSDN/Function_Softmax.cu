//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include "Function_Softmax.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"
#include "LSDN_mathfunctions.h"

template <typename T>
__global__ void kernel_get_max(const int num, const int dim, const T* data, T* out) {
	CUDA_KERNEL_LOOP(index, num) {
		T maxval = data[index*dim];
		for (int i = 1; i < dim; ++i) {
			maxval = max(data[index * dim + i], maxval);
		}
		out[index] = maxval;
	}
}

template <class N>
void SoftmaxFunction<N>::GetMax(i2t<true>) {
	kernel_get_max<ValueType><<<LSDN_GET_BLOCKS(numEl_AllButOne), LSDN_CUDA_NUM_THREADS>>>(int(numEl_AllButOne), int(NodeType::sz[0]), NodeType::value, scale_val);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_ElementwiseMultiply(const int num, const T* data_in1, const T* data_in2, T* data_out) {
	CUDA_KERNEL_LOOP(index, num) {
		data_out[index] = data_in1[index] * data_in2[index];
	}
}

template <typename T>
__global__ void kernel_ElementwiseMultiplyAdd(const int num, const T* data_in1, const T* data_in2, T* data_out) {
	CUDA_KERNEL_LOOP(index, num) {
		data_out[index] = data_in1[index] * data_in2[index];
	}
}

template <class N>
void SoftmaxFunction<N>::ElementwiseMultiply(i2t<true>, SizeType dim, ValueType* data_in1, ValueType* data_in2, ValueType* data_out, bool addToOutput) {
	if (addToOutput) {
		kernel_ElementwiseMultiplyAdd<ValueType><<<LSDN_GET_BLOCKS(numEl_AllButOne), LSDN_CUDA_NUM_THREADS>>>(int(dim), data_in1, data_in2, data_out);
	} else {
		kernel_ElementwiseMultiply<ValueType><<<LSDN_GET_BLOCKS(numEl_AllButOne), LSDN_CUDA_NUM_THREADS>>>(int(dim), data_in1, data_in2, data_out);
	}
	check_cuda_errors(__FILE__, __LINE__);
}

template class SoftmaxFunction<Node<double, int, false> >;
template class SoftmaxFunction<Node<double, int, true> >;
template class SoftmaxFunction<Node<float, int, false> >;
template class SoftmaxFunction<Node<float, int, true> >;