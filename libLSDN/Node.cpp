//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <iostream>
#include <assert.h>

#include "cuda_runtime.h"

#include "../LSDN_CudaCommon.h"

#include "Node.h"

template <class V, class S, bool G>
Node<V, S, G>::Node() : value(NULL), CPUDataOffset(NULL), sz(NULL), numDim(0), numEl(0), isMyValue(true) {}

template <class V, class S, bool G>
Node<V, S, G>::~Node() {}

template <class V, class S, bool G>
void Node<V, S, G>::SetName(const char* n) {
	name = n;
}

template <class V, class S, bool G>
void Node<V, S, G>::PrintName() {
	if (name.size())
		std::cout << name.c_str() << std::endl;
}

template <class V, class S, bool G>
void Node<V, S, G>::Clear()  {
	if (value != NULL && isMyValue) {
		DeAllocValueMem(GPUType(), value);
		value = NULL;
	}
	if (sz != NULL) {
		delete[] sz;
		sz = NULL;
	}
}

template <class V, class S, bool G>
void Node<V, S, G>::DeletedByContainer() {
	value = NULL;
}

template <class V, class S, bool G>
NODETYPE Node<V, S, G>::IdentifyMe()  {
	return NODE_UNDEF;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType* Node<V, S, G>::GetValuePtr() {
	return value;
}

template <class V, class S, bool G>
typename Node<V, S, G>::SizeType* Node<V, S, G>::GetSizePtr() {
	return sz;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType Node<V, S, G>::GetValue(size_t, int, int) {
	assert(false);
	return ValueType(0.0);
}

template <class V, class S, bool G>
typename Node<V, S, G>::SizeType Node<V, S, G>::GetNumDim() {
	return numDim;
}

template <class V, class S, bool G>
typename Node<V,S,G>::SizeType Node<V, S, G>::ComputeNumEl() {
	if (sz != NULL) {
		numEl = 1;
		for (SizeType k = 0; k < numDim; ++k) {
			numEl *= sz[k];
		}
	} else {
		numEl = 0;
	}
	return numEl;
}

template <class V, class S, bool G>
void Node<V, S, G>::AdjustDimension(TreePostIter&) {}

template <class V, class S, bool G>
void Node<V, S, G>::SetValueSize(ValueType* v, SizeType* s, SizeType dim, bool transferValueOwnership) {
	value = v;
	sz = s;
	numDim = dim;
	isMyValue = transferValueOwnership;
	ComputeNumEl();
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType** Node<V, S, G>::GetDiffGradientAndEmpMean() {
	return NULL;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType** Node<V, S, G>::GetDiffHist() {
	return NULL;
}

template <class V, class S, bool G>
typename Node<V, S, G>::SizeType Node<V, S, G>::GetNumEl() {
	return numEl;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType* Node<V, S, G>::AllocValueMem(i2t<true>, SizeType numEl) {
	ValueType* ptr = NULL;
	cudaMalloc((void**)&ptr, numEl*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	return ptr;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType* Node<V, S, G>::AllocValueMem(i2t<false>, SizeType numEl) {
	return new ValueType[numEl];
}

template <class V, class S, bool G>
void Node<V, S, G>::DeAllocValueMem(i2t<true>, ValueType* ptr) {
	cudaFree(ptr);
}

template <class V, class S, bool G>
void Node<V, S, G>::DeAllocValueMem(i2t<false>, ValueType* ptr) {
	delete[] ptr;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType* Node<V, S, G>::GetCPUDataOffset() {
	return CPUDataOffset;
}

template <class V, class S, bool G>
void Node<V, S, G>::SetCPUDataOffset(ValueType* val) {
	CPUDataOffset = val;
}

template <class V, class S, bool G>
typename Node<V, S, G>::ValueType* Node<V, S, G>::GetCPUDerivativeRootPtr() {
	return CPUDerivativeRootPtr;
}

template <class V, class S, bool G>
void Node<V, S, G>::SetCPUDerivativeRootPtr(ValueType* val) {
	CPUDerivativeRootPtr = val;
}


template class Node<double, int, false>;
template class Node<double, int, true>;
template class Node<float, int, false>;
template class Node<float, int, true>;