//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <random>
#include <iostream>
#include <string.h>

#include "cuda_runtime.h"

#include "LSDN.h"
#include "LSDN_mathfunctions.h"
#include "../LSDN_CudaCommon.h"

#include "Function_Dropout.h"

template <class N>
DropoutFunction<N>::DropoutFunction(const NodeParameters& params) : dropout_rate_(params.dropout_rate), scale_(ValueType(1.0) / (ValueType(1.0) - dropout_rate_)), mask_(NULL) {
	assert(dropout_rate_ > 0 && dropout_rate_ < 1);
}

template <class N>
DropoutFunction<N>::~DropoutFunction() {}

template <class N>
void DropoutFunction<N>::Clear() {
	if (mask_ != NULL){
		DeAllocMaskMem(typename NodeType::GPUType(), mask_);
		mask_ = NULL;
	}
	ComputeFunction<N>::Clear();
}

template <class N>
unsigned int* DropoutFunction<N>::AllocMaskMem(i2t<true>, SizeType numEl) {
	unsigned int* ptr = NULL;
	cudaMalloc((void**)&ptr, numEl*sizeof(unsigned int));
	check_cuda_errors(__FILE__, __LINE__);
	return ptr;
}

template <class N>
unsigned int* DropoutFunction<N>::AllocMaskMem(i2t<false>, SizeType numEl) {
	return new unsigned int[numEl];
}

template <class N>
void DropoutFunction<N>::DeAllocMaskMem(i2t<true>, unsigned int* ptr) {
	cudaFree(ptr);
}

template <class N>
void DropoutFunction<N>::DeAllocMaskMem(i2t<false>, unsigned int* ptr) {
	delete[] ptr;
}

template <class N>
void DropoutFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType numEl1 = c_b_ptr->GetNumEl();

	SizeType curNumEl = 1;
	SizeType negPos = -1;
	for (SizeType k = 0; k < NodeType::numDim; ++k) {
		curNumEl *= ((NodeType::sz[k] != -1) ? NodeType::sz[k] : 1);
		negPos = ((NodeType::sz[k] == -1) ? k : negPos);
	}

	if (negPos >= 0) {
		assert(numEl1%curNumEl == 0);
		NodeType::sz[negPos] = numEl1 / curNumEl;
	}
	NodeType::numEl = numEl1;
}

template <class N>
void DropoutFunction<N>::Evaluate(TreePostIter& cur, STATE state) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType childDimensions = c_b_ptr->GetNumDim();

	SizeType numEl1 = c_b_ptr->GetNumEl();

	if (NodeType::sz == NULL && NodeType::value == NULL){
		NodeType::numDim = childDimensions;
		NodeType::sz = new SizeType[NodeType::numDim];
		memcpy((char*)NodeType::sz, (char*)sz1, NodeType::numDim*sizeof(SizeType));
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl1);
		NodeType::numEl = numEl1;
	}
	assert(NodeType::numEl == numEl1);

	if (state == TRAIN){
		if (mask_ == NULL){
			mask_ = AllocMaskMem(typename NodeType::GPUType(), NodeType::numEl);
		}
		rngUniform(typename NodeType::GPUType(), NodeType::numEl, mask_);//ValueType(1) - dropout_rate_
		MaskOperation(typename NodeType::GPUType(), val1, NodeType::value, static_cast<unsigned int>(dropout_rate_*std::numeric_limits<unsigned int>::max()), NodeType::numEl, false);
	} else if (state == TRAIN_FUNEVAL) {
		assert(mask_ != NULL);
		MaskOperation(typename NodeType::GPUType(), val1, NodeType::value, static_cast<unsigned int>(dropout_rate_*std::numeric_limits<unsigned int>::max()), NodeType::numEl, false);
	} else if (state == TEST || state == VALIDATE) {
		LSDNMemCpy(typename NodeType::GPUType(), (char*)NodeType::value, (char*)val1, sizeof(ValueType)*NodeType::numEl);
	} else {
		assert(false);
	}
}

template <class N>
void DropoutFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	ValueType** output = c_b_ptr->GetDiffGradientAndEmpMean();

	assert(NodeType::numEl != 0);
	if (*output == NULL) {
		*output = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
#ifdef LSDN_USE_GRAPH
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output, 0, NodeType::numEl);
#endif
	}

	MaskOperation(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, *output, static_cast<unsigned int>(dropout_rate_*std::numeric_limits<unsigned int>::max()), NodeType::numEl, 
#ifdef LSDN_USE_GRAPH
		true
#else
		(c_b_ptr->IdentifyMe() == NODE_PARAM)
#endif
		);

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template <class N>
typename DropoutFunction<N>::ValueType DropoutFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template <class N>
void DropoutFunction<N>::rngUniform(i2t<false>, const SizeType numEl, unsigned int* mask) {
	std::uniform_int_distribution<unsigned int> distribution(0, std::numeric_limits<unsigned int>::max());
	auto f = std::bind(distribution, std::ref(LSDN::Instance().cpu_generator()));
	std::generate(mask, mask + numEl, f);
}

template <class N>
void DropoutFunction<N>::rngUniform(i2t<true>, const SizeType numEl, unsigned int* mask) {
	curandGenerate(LSDN::Instance().curand_generator(), mask, numEl);
	check_cuda_errors(__FILE__, __LINE__);
}

template <class N>
void DropoutFunction<N>::MaskOperation(i2t<false>, ValueType* input, ValueType* output, unsigned int threshold, SizeType Length, bool addToOutput) {
	if (addToOutput) {
		for (SizeType k = 0; k < Length; ++k){
			output[k] += input[k] * (mask_[k]>threshold) * scale_;
		}
	} else {
		for (SizeType k = 0; k < Length; ++k){
			output[k] = input[k] * (mask_[k]>threshold) * scale_;
		}
	}
}

template class DropoutFunction<Node<double, int, false> >;
template class DropoutFunction<Node<double, int, true> >;
template class DropoutFunction<Node<float, int, false> >;
template class DropoutFunction<Node<float, int, true> >;