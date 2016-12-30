//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include "ComputeFunction.h"

template <class N>
ComputeFunction<N>::ComputeFunction() : DiffGradNEmpMean(NULL) {}

template <class N>
ComputeFunction<N>::~ComputeFunction() {}

template <class N>
void ComputeFunction<N>::Clear() {
	if (DiffGradNEmpMean != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), DiffGradNEmpMean);
		DiffGradNEmpMean = NULL;
	}
	NodeType::Clear();
}

template <class N>
void ComputeFunction<N>::DeletedByContainer() {
	DiffGradNEmpMean = NULL;
	NodeType::value = NULL;
}

template <class N>
typename ComputeFunction<N>::ValueType** ComputeFunction<N>::GetDiffGradientAndEmpMean() {
	return &DiffGradNEmpMean;
}

template <class N>
void ComputeFunction<N>::Evaluate(TreePostIter&, STATE) {}

template <class N>
void ComputeFunction<N>::Gradient(TreePostIter&) {}

template class ComputeFunction<Node<double, int, false> >;
template class ComputeFunction<Node<double, int, true> >;
template class ComputeFunction<Node<float, int, false> >;
template class ComputeFunction<Node<float, int, true> >;