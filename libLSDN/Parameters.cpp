//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <iostream>
#include <assert.h>
#include "Parameters.h"

#include "LSDN_mathfunctions.h"

template <class N>
Parameters<N>::Parameters(const NodeParameters& params) : DiffGradNEmpMean(NULL), DiffHist(NULL), stepLength(params.stepLength), moment(params.momentum), stepDivider(params.stepDivider), l2_reg(params.l2_reg), norm_limit(params.norm_limit), UpdateCallback(params.UpdateCallback) {}

template <class N>
Parameters<N>::~Parameters() {}

template <class N>
NODETYPE Parameters<N>::IdentifyMe() {
	return NODE_PARAM;
}

template <class N>
void Parameters<N>::Clear() {
	if (DiffGradNEmpMean != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), DiffGradNEmpMean);
		DiffGradNEmpMean = NULL;
	}
	if (DiffHist != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), DiffHist);
		DiffHist = NULL;
	}
	NodeType::Clear();
}

template <class N>
void Parameters<N>::DeletedByContainer() {
	DiffGradNEmpMean = NULL;
	DiffHist = NULL;
	NodeType::value = NULL;
}

template <class N>
typename Parameters<N>::ValueType** Parameters<N>::GetDiffGradientAndEmpMean() {
	return &DiffGradNEmpMean;
}

template <class N>
typename Parameters<N>::ValueType** Parameters<N>::GetDiffHist() {
	return &DiffHist;
}

template <class N>
void Parameters<N>::Evaluate(TreePostIter&, STATE) {}

template <class N>
void Parameters<N>::Gradient(TreePostIter&) {}

template <class N>
void Parameters<N>::UpdateParameters() {
	if (moment != 0) {
		/*ScaleVecOrMat(typename NodeType::GPUType(), DiffHist, NodeType::GetNumEl(), moment);
		VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), ValueType(1.0), DiffGradNEmpMean, DiffHist);
		if (l2_reg != 0) {
			VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), l2_reg, NodeType::value, DiffHist);
		}
		VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), stepLength, DiffHist, NodeType::value);
		//VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), stepLength, DiffGradNEmpMean, NodeType::value);
		//VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), moment, DiffHist, NodeType::value);*/

		ScaleVecOrMat(typename NodeType::GPUType(), DiffHist, NodeType::GetNumEl(), moment);
		VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), stepLength, DiffGradNEmpMean, DiffHist);
		if (l2_reg != 0) {
			VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), l2_reg*stepLength, NodeType::value, DiffHist);
		}
		VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), ValueType(1.0), DiffHist, NodeType::value);
	} else {
		if (l2_reg != 0) {
			VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), l2_reg, NodeType::value, DiffGradNEmpMean);
		}
		VectorAdd(typename NodeType::GPUType(), NodeType::GetNumEl(), stepLength, DiffGradNEmpMean, NodeType::value);
	}
	if (norm_limit > 0) {
		assert(NodeType::numDim <= 2);
		NormLimitByCol(typename NodeType::GPUType(), NodeType::sz[0], ((NodeType::numDim == 1) ? 1 : NodeType::sz[1]), norm_limit, NodeType::value);
	}
	if (UpdateCallback != NULL) {
		(*UpdateCallback)(this);
	}
	//ValueType res = ValueType(0);
	//VectorInnerProduct(typename NodeType::GPUType(), NodeType::GetNumEl(), NodeType::value, NodeType::value, &res);
	//std::cout << " W: " << std::sqrt(res)/NodeType::GetNumEl() << std::endl;
}

template <class N>
void Parameters<N>::ReduceStepSize() {
	stepLength /= stepDivider;
}

template <class N>
typename Parameters<N>::ValueType Parameters<N>::GetRegularization() {
	if (l2_reg != 0) {
		ValueType res = 0;
		VectorInnerProduct(typename NodeType::GPUType(), NodeType::GetNumEl(), NodeType::value, NodeType::value, &res);
		return l2_reg*res / ValueType(2);
	}
	return 0;
}

template class Parameters<Node<double, int, false> >;
template class Parameters<Node<double, int, true> >;
template class Parameters<Node<float, int, false> >;
template class Parameters<Node<float, int, true> >;