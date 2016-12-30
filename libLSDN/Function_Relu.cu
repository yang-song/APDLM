//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <functional>

#include "cuda_runtime.h"

#include "Function_Relu.h"

#include "LSDN_mathfunctions.h"
#include "../LSDN_CudaCommon.h"

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n) {
		out[index] = in[index] > 0 ? in[index] : 0;
	}
}

template <class N>
void ReluFunction<N>::PerformReluForward(i2t<true>, ValueType* memIn, SizeType num, ValueType* memOut) {
	ReLUForward<ValueType><<<LSDN_GET_BLOCKS(num), LSDN_CUDA_NUM_THREADS>>>(int(num), memIn, memOut);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
	CUDA_KERNEL_LOOP(index, n) {
		out_diff[index] = in_diff[index] * (in_data[index] > 0);
	}
}
template <typename Dtype>
__global__ void ReLUBackwardAdd(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
	CUDA_KERNEL_LOOP(index, n) {
		out_diff[index] += in_diff[index] * (in_data[index] > 0);
	}
}

template <class N>
void ReluFunction<N>::PerformReluBackwardNoAdd(i2t<true>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut) {
	ReLUBackward<ValueType><<<LSDN_GET_BLOCKS(num), LSDN_CUDA_NUM_THREADS>>>(int(num), memIn, memCheckForLTZero, memOut);
}

template <class N>
void ReluFunction<N>::PerformReluBackwardAdd(i2t<true>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut) {
	ReLUBackwardAdd<ValueType><<<LSDN_GET_BLOCKS(num), LSDN_CUDA_NUM_THREADS>>>(int(num), memIn, memCheckForLTZero, memOut);
}

template <class N>
void ReluFunction<N>::Gradient(i2t<true>, TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	ValueType* val1 = c_b_ptr->GetValuePtr();
	ValueType** output = c_b_ptr->GetDiffGradientAndEmpMean();
	assert(output != NULL);

	assert(NodeType::numEl != 0);
	if (*output == NULL) {
		*output = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}

#ifdef LSDN_USE_GRAPH
	ReLUBackwardAdd<ValueType> << <LSDN_GET_BLOCKS(NodeType::numEl), LSDN_CUDA_NUM_THREADS >> >(int(NodeType::numEl), ComputeFunction<NodeType>::DiffGradNEmpMean, val1, *output);
#else
	if (c_b_ptr->IdentifyMe() == NODE_PARAM) {
		ReLUBackwardAdd<ValueType><<<LSDN_GET_BLOCKS(NodeType::numEl), LSDN_CUDA_NUM_THREADS>>>(int(NodeType::numEl), ComputeFunction<NodeType>::DiffGradNEmpMean, val1, *output);
	} else {
		ReLUBackward<ValueType><<<LSDN_GET_BLOCKS(NodeType::numEl), LSDN_CUDA_NUM_THREADS>>>(int(NodeType::numEl), ComputeFunction<NodeType>::DiffGradNEmpMean, val1, *output);
	}
#endif
	check_cuda_errors(__FILE__, __LINE__);

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template class ReluFunction<Node<double, int, false> >;
template class ReluFunction<Node<double, int, true> >;
template class ReluFunction<Node<float, int, false> >;
template class ReluFunction<Node<float, int, true> >;