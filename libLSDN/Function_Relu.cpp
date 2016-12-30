//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <assert.h>
#include <string.h>
#include <iostream>

#include "LSDN_mathfunctions.h"

#include "Function_Relu.h"

template <class N>
ReluFunction<N>::ReluFunction(const NodeParameters&) {}

template <class N>
ReluFunction<N>::~ReluFunction() {}

/*template <class N>
void ReluFunction<N>::Evaluate(TreePostIter cur, STATE state) {
	Evaluate(typename NodeType::GPUType(), cur, state);
}*/

template <class N>
void ReluFunction<N>::Gradient(TreePostIter& cur) {
	Gradient(typename NodeType::GPUType(), cur);
}

template <class N>
void ReluFunction<N>::AdjustDimension(TreePostIter& cur) {
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
void ReluFunction<N>::PerformReluForward(i2t<false>, ValueType* memIn, SizeType num, ValueType* memOut) {
	/*for (SizeType k = 0; k < num; ++k) {
		memOut[k] = std::max(memIn[k], ValueType(0));
	}*/
	auto f = std::bind((const ValueType &(*)(const ValueType&, const ValueType&))&std::max<ValueType>, std::placeholders::_1, ValueType(0));
	std::transform(memIn, memIn + num, memOut, f);
}

template <class N>
void ReluFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	SizeType numEl1 = c_b_ptr->GetNumEl();

	if (NodeType::sz == NULL && NodeType::value == NULL){
		NodeType::sz = new SizeType[numDim1];
		memcpy((char*)NodeType::sz, (char*)sz1, numDim1*sizeof(SizeType));
		NodeType::numDim = numDim1;
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl1);
		NodeType::numEl = numEl1;
	}
	assert(NodeType::numEl == numEl1);

	PerformReluForward(typename NodeType::GPUType(), val1, NodeType::numEl, NodeType::value);
}

template <class N>
void ReluFunction<N>::PerformReluBackwardNoAdd(i2t<false>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut) {
	auto f = [](const ValueType& a, const ValueType& b) -> ValueType {return a*(b > ValueType(0)); };//closure
	std::transform(memIn, memIn + num, memCheckForLTZero, memOut, f);
}

template <class N>
void ReluFunction<N>::PerformReluBackwardAdd(i2t<false>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut) {
	for (SizeType k = 0; k < num; ++k) {
		memOut[k] += memIn[k] * (memCheckForLTZero[k]>ValueType(0.0));
	}
}

template <class N>
void ReluFunction<N>::Gradient(i2t<false>, TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	ValueType* val1 = c_b_ptr->GetValuePtr();
	ValueType** output = c_b_ptr->GetDiffGradientAndEmpMean();
	assert(output != NULL);

	assert(NodeType::numEl != 0);
	if (*output == NULL) {
		*output = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
#ifdef LSDN_USE_GRAPH
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output, 0, NodeType::numEl);
#endif
	}

#ifdef LSDN_USE_GRAPH
	PerformReluBackwardAdd(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, val1, NodeType::numEl, *output);
#else
	if (c_b_ptr->IdentifyMe() == NODE_PARAM) {
		/*ValueType* buf = *output;
		for (SizeType k = 0, k_e = NodeType::numEl; k < k_e; ++k) {
			buf[k] += ComputeFunction<NodeType>::DiffGradNEmpMean[k] * (val1[k]>ValueType(0.0));
		}*/
		PerformReluBackwardAdd(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, val1, NodeType::numEl, *output);
	} else {
		//auto f = [](const ValueType& a, const ValueType& b) -> ValueType {return a*(b > ValueType(0)); };//closure
		//std::transform(ComputeFunction<NodeType>::DiffGradNEmpMean, ComputeFunction<NodeType>::DiffGradNEmpMean + NodeType::numEl, val1, *output, f);
		PerformReluBackwardNoAdd(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, val1, NodeType::numEl, *output);
	}
#endif

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template <class N>
typename ReluFunction<N>::ValueType ReluFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template class ReluFunction<Node<double, int, false> >;
template class ReluFunction<Node<double, int, true> >;
template class ReluFunction<Node<float, int, false> >;
template class ReluFunction<Node<float, int, true> >;