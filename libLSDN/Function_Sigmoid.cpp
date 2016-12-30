//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <algorithm>
#include <assert.h>
#include <string.h>

#include "LSDN_mathfunctions.h"

#include "Function_Sigmoid.h"

template <class N>
SigmoidFunction<N>::SigmoidFunction(const NodeParameters&) {}

template <class N>
SigmoidFunction<N>::~SigmoidFunction() {}

template <class N>
void SigmoidFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim_input = c_b_ptr->GetNumDim();

	if (NodeType::sz == NULL && NodeType::value == NULL && NodeType::numEl == 0){
		NodeType::numEl = c_b_ptr->GetNumEl();

		NodeType::sz = new SizeType[numDim_input];
		memcpy((char*)NodeType::sz, (char*)sz1, numDim_input*sizeof(SizeType));
		NodeType::numDim = numDim_input;
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}

	SigmoidForward(typename NodeType::GPUType(), val1);
}

template <class N>
void SigmoidFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	//SizeType* sz1 = (*c_b)->GetSizePtr();
	//ValueType* val1 = (*c_b)->GetValuePtr();
	ValueType** output = c_b_ptr->GetDiffGradientAndEmpMean();
	assert(output != NULL);

	if (*output == NULL) {
		*output = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
#ifdef LSDN_USE_GRAPH
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output, 0, NodeType::numEl);
#endif
	}

	SigmoidBackward(typename NodeType::GPUType(), *output, 
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
void SigmoidFunction<N>::SigmoidForward(i2t<false>, ValueType* input) {
	auto f = [](const ValueType& x) -> ValueType{ return ValueType(1.) / (ValueType(1.) + std::exp(-x)); };
	std::transform(input, input + NodeType::numEl, NodeType::value, f);
}

template <class N>
void SigmoidFunction<N>::SigmoidBackward(i2t<false>, ValueType* output, bool addToOutput) {
	if (addToOutput) {
		for (SizeType k = 0; k < NodeType::numEl; ++k) {
			ValueType sigmoid_x = NodeType::value[k];
			output[k] += ComputeFunction<NodeType>::DiffGradNEmpMean[k] * sigmoid_x*(ValueType(1.0) - sigmoid_x);
		}
	} else {
		auto f = [](const ValueType& x, const ValueType& y) -> ValueType{ return x*y*(ValueType(1.) - y); };
		std::transform(ComputeFunction<NodeType>::DiffGradNEmpMean, ComputeFunction<NodeType>::DiffGradNEmpMean + NodeType::numEl, NodeType::value, output, f);
	}
}

template <class N>
typename SigmoidFunction<N>::ValueType SigmoidFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template class SigmoidFunction<Node<double, int, false> >;
template class SigmoidFunction<Node<double, int, true> >;
template class SigmoidFunction<Node<float, int, false> >;
template class SigmoidFunction<Node<float, int, true> >;
