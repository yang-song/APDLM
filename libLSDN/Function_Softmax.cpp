//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <algorithm>
#include <functional>
#include <string.h>
#include <assert.h>

#include "Function_Softmax.h"

#include "LSDN.h"
#include "LSDN_mathfunctions.h"

template <class N>
SoftmaxFunction<N>::SoftmaxFunction(const NodeParameters&) : scale_val(NULL), all_one(NULL), numEl_AllButOne(0) {}

template <class N>
SoftmaxFunction<N>::~SoftmaxFunction() {}

template <class N>
void SoftmaxFunction<N>::Clear() {
	if (scale_val != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), scale_val);
		scale_val = NULL;
	}
	if (all_one != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), all_one);
		all_one = NULL;
	}
	ComputeFunction<N>::Clear();
}

template <class N>
void SoftmaxFunction<N>::Evaluate(TreePostIter& cur, STATE){
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	//typename tree<NodeType*>::sibling_iterator c_b = cur.begin();
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();

	if (NodeType::sz == NULL && NodeType::value == NULL && NodeType::numEl == 0){
		NodeType::numEl = c_b_ptr->GetNumEl();
		numEl_AllButOne = NodeType::numEl / sz1[0];
		NodeType::sz = new SizeType[numDim1];
		memcpy((char*)NodeType::sz, (char*)sz1, sizeof(SizeType)*numDim1);
		NodeType::numDim = numDim1;
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}

	LSDNMemCpy(typename NodeType::GPUType(), NodeType::value, val1, sizeof(ValueType)*NodeType::numEl);

	//find the max accross dimension 0
	if (scale_val == NULL) {
		scale_val = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_AllButOne);
	}
	GetMax(typename NodeType::GPUType());

	if (all_one == NULL) {
		all_one = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::sz[0]);
		LSDNMemSet(typename NodeType::GPUType(), all_one, ValueType(1.0), NodeType::sz[0]);
	}

	MultiplyMatMat(typename NodeType::GPUType(), all_one, scale_val, NodeType::value, NodeType::sz[0], numEl_AllButOne, 1, CblasNoTrans, CblasNoTrans, -1.0, 1.0);

	ElementwiseExp(typename NodeType::GPUType(), NodeType::value, NodeType::value, size_t(NodeType::numEl));

	MultiplyMatVec(typename NodeType::GPUType(), NodeType::value, all_one, scale_val, NodeType::sz[0], numEl_AllButOne, CblasTrans, ValueType(1.0), ValueType(0.0));

	for (SizeType n = 0; n < numEl_AllButOne; ++n){
		ScaleVecOrMat(typename NodeType::GPUType(), NodeType::value + n*NodeType::sz[0], NodeType::sz[0], ValueType(1.0) / scale_val[n]);
	}
}

template <class N>
void SoftmaxFunction<N>::GetMax(i2t<false>) {
	for (SizeType n = 0; n < numEl_AllButOne; ++n){
		scale_val[n] = NodeType::value[n*NodeType::sz[0]];

		for (SizeType m = 1; m < NodeType::sz[0]; ++m){
			scale_val[n] = std::max(scale_val[n], NodeType::value[m + n*NodeType::sz[0]]);
		}
	}
}

template <class N>
void SoftmaxFunction<N>::DiagonalInnerProduct(i2t<false>) {
	for (SizeType n = 0; n < numEl_AllButOne; ++n){
		VectorInnerProduct(typename NodeType::GPUType(), NodeType::sz[0], ComputeFunction<NodeType>::DiffGradNEmpMean + n * NodeType::sz[0], NodeType::value + n * NodeType::sz[0], &scale_val[n]);
	}
}

//could be done with cublas device api which requires compute capability 3.5
template <class N>
void SoftmaxFunction<N>::DiagonalInnerProduct(i2t<true>) {
	cublasSetPointerMode(LSDN::Instance().cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
	for (SizeType n = 0; n < numEl_AllButOne; ++n) {
		VectorInnerProduct(typename NodeType::GPUType(), NodeType::sz[0], ComputeFunction<NodeType>::DiffGradNEmpMean + n * NodeType::sz[0], NodeType::value + n * NodeType::sz[0], &scale_val[n]);
	}
	cublasSetPointerMode(LSDN::Instance().cublas_handle(), CUBLAS_POINTER_MODE_HOST);
}

template <class N>
void SoftmaxFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	ValueType** output = c_b_ptr->GetDiffGradientAndEmpMean();
	assert(output != NULL);
	if (*output == NULL) {
		*output = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
#ifdef LSDN_USE_GRAPH
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output, 0, NodeType::numEl);
#endif
	}

	//goal: compute (diag(y)-yy^T)v = v.*y - (y^Tv).*y = y.*(v - (y^Tv)*1), where 1 is a column vector
	if (scale_val == NULL){
		assert(false);//something would be wrong...
		//scale_val = new ValueType[numEl_AllButOne];
	}

	//compute (y^Tv) for each sample
	DiagonalInnerProduct(typename NodeType::GPUType());

	//compute (v - (y^Tv)*1)
	if (all_one == NULL) {
		assert(false);//something would be wrong...
		//all_one = new ValueType[sz[0]];
		//std::fill(all_one, all_one + sz[0], ValueType(1.0));
	}

	if (
#ifdef LSDN_USE_GRAPH
		true
#else
		c_b_ptr->IdentifyMe() == NODE_PARAM
#endif
		) {
		ValueType* tmp_buffer = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
		LSDNMemCpy(typename NodeType::GPUType(), tmp_buffer, ComputeFunction<NodeType>::DiffGradNEmpMean, sizeof(ValueType)*NodeType::numEl);

		MultiplyMatMat(typename NodeType::GPUType(), all_one, scale_val, tmp_buffer, NodeType::sz[0], numEl_AllButOne, 1, CblasNoTrans, CblasNoTrans, -1.0, 1.0);

		//compute y.*(v - (y^Tv)*1)
		ElementwiseMultiply(typename NodeType::GPUType(), NodeType::numEl, NodeType::value, tmp_buffer, *output, true);
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), tmp_buffer);
	} else {
		//memcpy((char*)*output, (char*)ComputeFunction<NodeType>::DiffGradNEmpMean, sizeof(ValueType)*NodeType::numEl);
		//std::copy(DiffGradNEmpMean, DiffGradNEmpMean + numEl, *output);
		LSDNMemCpy(typename NodeType::GPUType(), *output, ComputeFunction<NodeType>::DiffGradNEmpMean, sizeof(ValueType)*NodeType::numEl);

		MultiplyMatMat(typename NodeType::GPUType(), all_one, scale_val, *output, NodeType::sz[0], numEl_AllButOne, 1, CblasNoTrans, CblasNoTrans, -1.0, 1.0);

		//compute y.*(v - (y^Tv)*1)
		ElementwiseMultiply(typename NodeType::GPUType(), NodeType::numEl, NodeType::value, *output, *output, false);
	}

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template <class N>
void SoftmaxFunction<N>::ElementwiseMultiply(i2t<false>, SizeType dim, ValueType* data_in1, ValueType* data_in2, ValueType* data_out, bool addToOutput) {
	if (addToOutput) {
		for (SizeType k = 0; k < dim; ++k){
			data_out[k] += data_in1[k] * data_in2[k];
		}
	} else {
		std::transform(data_in1, data_in1 + dim, data_in2, data_out, std::multiplies<ValueType>());
	}
}

template <class N>
typename SoftmaxFunction<N>::ValueType SoftmaxFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template class SoftmaxFunction<Node<double, int, false> >;
template class SoftmaxFunction<Node<double, int, true> >;
template class SoftmaxFunction<Node<float, int, false> >;
template class SoftmaxFunction<Node<float, int, true> >;