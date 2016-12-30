//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif

#include <iostream>
#include <functional>
#include <fstream>
#include <assert.h>
#include <string.h>

#include "Function_Affine.h"
#include "Function_Relu.h"

#include "LSDN_mathfunctions.h"

#include "cuda_runtime.h"

template <class N>
AffineFunction<N>::AffineFunction(const NodeParameters& params) : hasBias_(params.hasBias), hasRelu_(params.hasRelu) {}

template <class N>
AffineFunction<N>::~AffineFunction() {}

template <class N>
void AffineFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() > 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	
	SizeType* sz_w = c_b_ptr->GetSizePtr();
	SizeType numDim_w = c_b_ptr->GetNumDim();
	assert(numDim_w <= 2);

	++c_b;
	c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	SizeType* sz_phi = c_b_ptr->GetSizePtr();
	SizeType numDim_phi = c_b_ptr->GetNumDim();

	SizeType NumberOutputElements = 1;
	if (numDim_w == 2) {
		NumberOutputElements = sz_w[1];
	}
	SizeType cnter = 1;//check how many dimensions of phi we need to fit sz_w[0]
	SizeType buffer = sz_phi[0];
	for (; cnter < numDim_phi && buffer != sz_w[0]; ++cnter) {
		buffer *= sz_phi[cnter];
	}
	for (SizeType k = cnter; k < numDim_phi; ++k) {
		NumberOutputElements *= sz_phi[k];
	}
	assert(NumberOutputElements>0);

	SizeType curNumEl = 1;
	SizeType negPos = -1;
	for (SizeType k = 0; k < NodeType::numDim; ++k) {
		curNumEl *= ((NodeType::sz[k] != -1) ? NodeType::sz[k] : 1);
		negPos = ((NodeType::sz[k] == -1) ? k : negPos);
	}
	if (negPos >= 0) {
		SizeType missingDim = NumberOutputElements / curNumEl;
		assert(NumberOutputElements%curNumEl == 0);
		NodeType::sz[negPos] = missingDim;
	}

	NodeType::numEl = NumberOutputElements;
}

template <class N>
void AffineFunction<N>::AdditionModuloOperand(i2t<false>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo) {
	for (SizeType k = 0; k < numEl; ++k) {
		SizeType ix = (k / op_division) % op_modulo;
		res[k] += addend[ix];
	}
}

template <class N>
void AffineFunction<N>::BiasDerivativeSingleDim(i2t<false>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition) {
	if (performAddition) {
		for (SizeType ix = 0; ix < sampleSize; ++ix) {
			ValueType* ptr = input + ix;
			for (SizeType k = 0; k < numSamples; ++k) {
				res[ix] += ptr[k*sampleSize];
			}
		}
	} else {
		for (SizeType ix = 0; ix < sampleSize; ++ix) {
			ValueType* ptr = input + ix;
			res[ix] = ValueType(0);
			for (SizeType k = 0; k < numSamples; ++k) {
				res[ix] += ptr[k*sampleSize];
			}
		}
	}
}

template <class N>
void AffineFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() > 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz_w = c_b_ptr->GetSizePtr();
	ValueType* val_w = c_b_ptr->GetValuePtr();
	SizeType numDim_w = c_b_ptr->GetNumDim();
	assert(numDim_w <= 2);

	++c_b;
	c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	SizeType* sz_phi = c_b_ptr->GetSizePtr();
	ValueType* val_phi = c_b_ptr->GetValuePtr();
	SizeType numDim_phi = c_b_ptr->GetNumDim();

	/*int numCols = sz_phi[numDim_phi - 1];
	if (numDim_w == 1) {//added
		for (SizeType k = 1; k < numDim_phi - 1; ++k) {
			numCols *= sz_phi[k];
		}
	}*/
	SizeType cnter = 1;
	SizeType buffer = sz_phi[0];
	for (; cnter < numDim_phi && buffer!=sz_w[0]; ++cnter) {
		buffer *= sz_phi[cnter];
	}
	SizeType numCols = c_b_ptr->GetNumEl() / buffer;

	assert(sz_w[0] == c_b_ptr->GetNumEl() / numCols);

	if (NodeType::sz == NULL && NodeType::value == NULL) {
		if (numDim_w == 2) {
			NodeType::sz = new SizeType[1 + numDim_phi - cnter];
			NodeType::sz[0] = sz_w[1];
			memcpy((char*)(NodeType::sz + 1), (char*)(sz_phi + cnter), sizeof(SizeType)*(numDim_phi - cnter));
			NodeType::numDim = 1 + numDim_phi - cnter;

			/*NodeType::sz = new SizeType[numDim_phi];
			NodeType::sz[0] = sz_w[1];
			memcpy((char*)(NodeType::sz + 1), (char*)(sz_phi + 1), sizeof(SizeType)*(numDim_phi - 1));
			NodeType::numDim = numDim_phi;*/
		} else {
			NodeType::sz = new SizeType[numDim_phi - 1];
			memcpy((char*)NodeType::sz, (char*)(sz_phi + 1), sizeof(SizeType)*(numDim_phi - 1));
			NodeType::numDim = numDim_phi - 1;
		}
		NodeType::ComputeNumEl();
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}
	assert(NodeType::numEl != 0);

	LSDNMemSet<ValueType>(typename NodeType::GPUType(), NodeType::value, 0, NodeType::numEl);
	
	MultiplyMatMat(typename NodeType::GPUType(), val_w, val_phi, NodeType::value, ((numDim_w == 1) ? 1 : sz_w[1]), numCols, sz_w[0], CblasTrans, CblasNoTrans);

	++c_b;
	if (hasBias_) {
		assert(c_b != cur.end());
		assert(NodeType::numDim == 2);
		c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
		ValueType* val_next = c_b_ptr->GetValuePtr();
		SizeType* sz_next = c_b_ptr->GetSizePtr();
		SizeType numDim_next = c_b_ptr->GetNumDim();
		assert(numDim_next == 1 && sz_next[0] == NodeType::sz[0]);
		AdditionModuloOperand(typename NodeType::GPUType(), NodeType::value, NodeType::numEl, val_next, SizeType(1), sz_next[0]);
		++c_b;
	}

	for (typename NodeType::TreeSiblIter ptr_e = cur.end(); c_b != ptr_e; ++c_b) {
		c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
		ValueType* val_next = c_b_ptr->GetValuePtr();
		SizeType numEl_next = c_b_ptr->GetNumEl();
		assert(NodeType::numEl == numEl_next);
		VectorAdd(typename NodeType::GPUType(), numEl_next, ValueType(1.0), val_next, NodeType::value);
	}

	if (hasRelu_) {
		ReluFunction<N>::PerformReluForward(typename NodeType::GPUType(), NodeType::value, NodeType::numEl, NodeType::value);
	}
}

template <class N>
void AffineFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() > 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz_w = c_b_ptr->GetSizePtr();
	SizeType numEl_w = c_b_ptr->GetNumEl();
	ValueType* val_w = c_b_ptr->GetValuePtr();
	SizeType numDim_w = c_b_ptr->GetNumDim();
	ValueType** output_w = c_b_ptr->GetDiffGradientAndEmpMean();
#ifndef LSDN_USE_GRAPH
	NODETYPE type_w = c_b_ptr->IdentifyMe();
#endif
	//assert(output_w != NULL);

	++c_b;
	c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	SizeType* sz_phi = c_b_ptr->GetSizePtr();
	SizeType numEl_phi = c_b_ptr->GetNumEl();
	ValueType* val_phi = c_b_ptr->GetValuePtr();
	SizeType numDim_phi = c_b_ptr->GetNumDim();
	ValueType** output_phi = c_b_ptr->GetDiffGradientAndEmpMean();

	if (hasRelu_) {
		ReluFunction<N>::PerformReluBackwardNoAdd(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::value, NodeType::numEl, ComputeFunction<NodeType>::DiffGradNEmpMean);
	}

	/*int numCols = 1;
	for (SizeType k = 1; k < numDim_phi; ++k) {
	numCols *= sz_phi[k];
	}*/
	/*int numCols = sz_phi[numDim_phi - 1];
	if (numDim_w == 1) {//added
	for (SizeType k = 1; k < numDim_phi - 1; ++k) {
	numCols *= sz_phi[k];
	}
	}*/
	SizeType cnter = 1;
	SizeType buffer = sz_phi[0];
	for (; cnter < numDim_phi && buffer != sz_w[0]; ++cnter) {
		buffer *= sz_phi[cnter];
	}
	SizeType numCols = c_b_ptr->GetNumEl() / buffer;

	//diff w.r.t. first input
	if (output_w != NULL) {
		if (*output_w == NULL) {
			*output_w = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_w);
			LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_w, 0, numEl_w);
		}
#ifndef LSDN_USE_GRAPH
		else if (type_w != NODE_PARAM) {
			LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_w, 0, numEl_w);
			//std::fill(*output_w, *output_w + numEl_w, ValueType(0.0));
		}
#endif

		//MultiplyMatMat(typename NodeType::GPUType(), val_phi, ComputeFunction<NodeType>::DiffGradNEmpMean, *output_w, sz_phi[0], ((numDim_w == 1) ? 1 : sz_w[1]), numCols, CblasNoTrans, CblasTrans);
		MultiplyMatMat(typename NodeType::GPUType(), val_phi, ComputeFunction<NodeType>::DiffGradNEmpMean, *output_w, sz_w[0], ((numDim_w == 1) ? 1 : sz_w[1]), numCols, CblasNoTrans, CblasTrans);
	}

	//diff w.r.t. second input
	if (output_phi != NULL) {
		if (*output_phi == NULL) {
			*output_phi = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_phi);
			LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_phi, 0, numEl_phi);
		}
#ifndef LSDN_USE_GRAPH
		else if (c_b_ptr->IdentifyMe() != NODE_PARAM) {
			LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_phi, 0, numEl_phi);
			//std::fill(*output_phi, *output_phi + numEl_phi, ValueType(0.0));
		}
#endif
		MultiplyMatMat(typename NodeType::GPUType(), val_w, ComputeFunction<NodeType>::DiffGradNEmpMean, *output_phi, sz_w[0], numCols, ((numDim_w == 1) ? 1 : sz_w[1]), CblasNoTrans, CblasNoTrans);
	}

	//diff w.r.t. remaining inputs
	++c_b;
	if (hasBias_) {
		assert(c_b != cur.end());
		assert(NodeType::numDim == 2);
		c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
		ValueType** output_next = c_b_ptr->GetDiffGradientAndEmpMean();
		SizeType numEl_next = c_b_ptr->GetNumEl();
		if (output_next != NULL) {
			if (*output_next == NULL) {
				*output_next = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_next);
#ifdef LSDN_USE_GRAPH
				LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_next, 0, numEl_next);
#endif
			}
#ifndef LSDN_USE_GRAPH
			//else if (c_b_ptr->IdentifyMe() != NODE_PARAM) {
			//	LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_next, 0, numEl_next);
			//}
#endif

			SizeType* sz_next = c_b_ptr->GetSizePtr();
			SizeType numDim_next = c_b_ptr->GetNumDim();
			assert(numDim_next == 1 && sz_next[0] == NodeType::sz[0]);
			BiasDerivativeSingleDim(typename NodeType::GPUType(), *output_next, ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::sz[0], NodeType::sz[1], 
#ifdef LSDN_USE_GRAPH
				true
#else
				(c_b_ptr->IdentifyMe() == NODE_PARAM)
#endif
				);
		}
		++c_b;
	}

	for (typename NodeType::TreeSiblIter ptr_e = cur.end(); c_b != ptr_e; ++c_b) {
		c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
		ValueType** output_next = c_b_ptr->GetDiffGradientAndEmpMean();
		if (output_next != NULL) {
			SizeType numEl_next = c_b_ptr->GetNumEl();
			if (*output_next == NULL) {
				*output_next = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_next);
#ifdef LSDN_USE_GRAPH
				LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_next, 0, numEl_next);
#endif
			}

#ifdef LSDN_USE_GRAPH
			VectorAdd(typename NodeType::GPUType(), numEl_next, ValueType(1.0), ComputeFunction<NodeType>::DiffGradNEmpMean, *output_next);
#else
			if (c_b_ptr->IdentifyMe() != NODE_PARAM) {
				LSDNMemCpy(typename NodeType::GPUType(), *output_next, ComputeFunction<NodeType>::DiffGradNEmpMean, sizeof(ValueType)*numEl_next);
			} else {
				VectorAdd(typename NodeType::GPUType(), numEl_next, ValueType(1.0), ComputeFunction<NodeType>::DiffGradNEmpMean, *output_next);
			}
#endif
		}
	}

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template <class N>
typename AffineFunction<N>::ValueType AffineFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template class AffineFunction<Node<double, int, false> >;
template class AffineFunction<Node<double, int, true> >;
template class AffineFunction<Node<float, int, false> >;
template class AffineFunction<Node<float, int, true> >;