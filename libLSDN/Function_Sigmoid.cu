//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include "Function_Sigmoid.h"

#include "cuda_runtime.h"

#include "../LSDN_CudaCommon.h"

template <typename T>
__global__ void SigmoidForwardKernel(const int n, const T* in, T* out) {
	CUDA_KERNEL_LOOP(index, n) {
		out[index] = 1. / (1. + exp(-in[index]));
	}
}

template <class N>
void SigmoidFunction<N>::SigmoidForward(i2t<true>, ValueType* input) {
	SigmoidForwardKernel<ValueType><<<LSDN_GET_BLOCKS(NodeType::numEl), LSDN_CUDA_NUM_THREADS>>>(int(NodeType::numEl), input, NodeType::value);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void SigmoidBackwardKernel(const int n, const T* in_diff, const T* out_data, T* out_diff) {
	CUDA_KERNEL_LOOP(index, n) {
		const T sigmoid_x = out_data[index];
		out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
	}
}

template <typename T>
__global__ void SigmoidBackwardAddKernel(const int n, const T* in_diff, const T* out_data, T* out_diff) {
	CUDA_KERNEL_LOOP(index, n) {
		const T sigmoid_x = out_data[index];
		out_diff[index] += in_diff[index] * sigmoid_x * (1 - sigmoid_x);
	}
}

template <class N>
void SigmoidFunction<N>::SigmoidBackward(i2t<true>, ValueType* output, bool addToOutput) {
	if (addToOutput) {
		SigmoidBackwardAddKernel<ValueType><<<LSDN_GET_BLOCKS(NodeType::numEl), LSDN_CUDA_NUM_THREADS>>>(int(NodeType::numEl), ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::value, output);
	} else {
		SigmoidBackwardKernel<ValueType><<<LSDN_GET_BLOCKS(NodeType::numEl), LSDN_CUDA_NUM_THREADS>>>(int(NodeType::numEl), ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::value, output);
	}
	check_cuda_errors(__FILE__, __LINE__);
}

template class SigmoidFunction<Node<double, int, false> >;
template class SigmoidFunction<Node<double, int, true> >;
template class SigmoidFunction<Node<float, int, false> >;
template class SigmoidFunction<Node<float, int, true> >;