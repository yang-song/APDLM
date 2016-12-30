//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif

#include <assert.h>
#include <string.h>
#include <iostream>

#include "Function_Conv.h"
#include "Function_Relu.h"

#include "LSDN_mathfunctions.h"

template <class N>
ConvFunction<N>::ConvFunction(const NodeParameters& params) : patchMatrix_(NULL), patchMatrixDiff_(NULL), sz_pm_(NULL), padSize_(params.p), stride_(params.s), numGroup_(params.g), hasBias_(params.hasBias), hasRELU_(params.hasRELU) {}

template <class N>
ConvFunction<N>::~ConvFunction() {}

template <class N>
void ConvFunction<N>::Clear() {
	if (patchMatrix_ != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), patchMatrix_);
		patchMatrix_ = NULL;
	}
	if (patchMatrixDiff_ != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), patchMatrixDiff_);
		patchMatrixDiff_ = NULL;
	}
	if (sz_pm_ != NULL) {
		delete[] sz_pm_;
		sz_pm_ = NULL;
	}

	ComputeFunction<N>::Clear();
}

template <class N>
void ConvFunction<N>::AccumPatchDiff2Im(i2t<false>, ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol) {
	//accumulated the gradients of extracted patches back to img

	//memset(img, 0, sizeof(ValueType)*numRow*numCol*numChan);

	SizeType h_out = (numRow + 2*padSize_ - kNumRow) / stride_ + 1;
	SizeType w_out = (numCol + 2*padSize_ - kNumCol) / stride_ + 1;
	SizeType c_out = numChan * kNumRow * kNumCol;

	//c is the position of each patch (column-order); e.g., c=0 is the top left-most position of each patch
	for(SizeType c = 0; c < c_out; ++c) {
		//column order
		//assert(false);//to be checked
		SizeType h_offset = c % kNumRow;
		SizeType w_offset = (c / kNumRow) % kNumCol;
		SizeType c_img = c / kNumRow / kNumCol;

		for(SizeType h = 0; h < h_out; ++h) {
			SizeType h_pad = h * stride_ - padSize_ + h_offset;

			//each row is a col-vector of patch
			for(SizeType w = 0; w < w_out; ++w) {
				SizeType w_pad = w * stride_ - padSize_ + w_offset;

				if (h_pad >= 0 && h_pad < numRow && w_pad >= 0 && w_pad < numCol) {
					img[(c_img * numCol + w_pad) * numRow + h_pad] += patchMatrixDiff_[(c*w_out+w)*h_out+h];
				}
			}
		}
	}
}

template <class N>
void ConvFunction<N>::Im2Patches(i2t<false>, ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol) {
	//extract patches of size kNumRow-kNumCol-numChannel from img, and concatenate them into a matrix
	//each row stores the column vectorized of each patch

	SizeType h_out = (numRow + 2*padSize_ - kNumRow) / stride_ + 1;
	SizeType w_out = (numCol + 2*padSize_ - kNumCol) / stride_ + 1;
	SizeType c_out = numChan * kNumRow * kNumCol;

	//c is the position of each patch; e.g., c=0 is the top left-most position of each patch
	for(SizeType c = 0; c < c_out; ++c) {
		//column order
		SizeType h_offset = c % kNumRow;
		SizeType w_offset = (c / kNumRow) % kNumCol;
		SizeType c_img = c / kNumRow / kNumCol;

		for(SizeType h = 0; h < h_out; ++h) {
			SizeType h_pad = h * stride_ - padSize_ + h_offset;

			//each row is a col-vector of patch
			for(SizeType w = 0; w < w_out; ++w) {
				SizeType w_pad = w * stride_ - padSize_ + w_offset;

				if (h_pad >= 0 && h_pad < numRow && w_pad >= 0 && w_pad < numCol) {
					patchMatrix_[(c*w_out+w)*h_out+h] = img[(c_img * numCol + w_pad) * numRow + h_pad];
				} else {
					patchMatrix_[(c*w_out+w)*h_out+h] = 0;
				}

			}
		}
	}
}

template <class N>
void ConvFunction<N>::AdditionModuloOperand(i2t<false>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo) {
	for (SizeType k = 0; k < numEl; ++k) {
		SizeType ix = (k / op_division) % op_modulo;
		res[k] += addend[ix];
	}
}

template <class N>
void ConvFunction<N>::BiasDerivativeSingleDim(i2t<false>, ValueType* res, ValueType* input, SizeType patchSize, SizeType numSamples, SizeType numChannels, bool performAddition) {
	if (performAddition) {
		SizeType k_e = patchSize * numSamples;
		SizeType sampleSize = patchSize*numChannels;
		for (SizeType ix = 0; ix < numChannels; ++ix) {
			ValueType* ptr = input + patchSize*ix;
			for (SizeType k = 0; k < k_e; ++k) {
				SizeType sample = k / patchSize;
				SizeType offset = k % patchSize;
				res[ix] += ptr[sampleSize*sample + offset];
			}
		}
	} else {
		SizeType k_e = patchSize * numSamples;
		SizeType sampleSize = patchSize*numChannels;
		for (SizeType ix = 0; ix < numChannels; ++ix) {
			ValueType* ptr = input + patchSize*ix;
			res[ix] = ValueType(0);
			for (SizeType k = 0; k < k_e; ++k) {
				SizeType sample = k / patchSize;
				SizeType offset = k % patchSize;
				res[ix] += ptr[sampleSize*sample + offset];
			}
		}
	}
}

template <class N>
void ConvFunction<N>::BiasDerivativeMultiDim(i2t<false>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition) {
	if (performAddition) {
		for (SizeType ix = 0; ix < sampleSize; ++ix) {
			for (SizeType k = 0; k < numSamples; ++k) {
				res[ix] += input[k*sampleSize + ix];
			}
		}
	} else {
		for (SizeType ix = 0; ix < sampleSize; ++ix) {
			res[ix] = ValueType(0);
			for (SizeType k = 0; k < numSamples; ++k) {
				res[ix] += input[k*sampleSize + ix];
			}
		}
	}
}

template <class N>
void ConvFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() >(hasBias_ ? 2ul : 1ul));
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz_w = c_b_ptr->GetSizePtr();
	SizeType numDim_w = c_b_ptr->GetNumDim();
	assert(numDim_w == 4);
	//assume sz_w: h, w, channel, num_filters

	++c_b;
	c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	SizeType* sz_phi = c_b_ptr->GetSizePtr();
	SizeType numDim_phi = c_b_ptr->GetNumDim();
	assert(numDim_phi == 4);
	
	assert(NodeType::numDim == 4);

	if (NodeType::sz[3] == -1) {
		NodeType::sz[3] = sz_phi[3];
		NodeType::ComputeNumEl();
	}

	SizeType sz_exp[4]{ (sz_phi[0] + 2 * padSize_ - sz_w[0]) / stride_ + 1, (sz_phi[1] + 2 * padSize_ - sz_w[1]) / stride_ + 1, sz_w[3], sz_phi[3] };
	if (NodeType::sz[0] != sz_exp[0] || NodeType::sz[1] != sz_exp[1] || NodeType::sz[2] != sz_exp[2] || NodeType::sz[3] != sz_exp[3]) {
		std::cout << "Dimensions don't fit. Got (";
		for (int k = 0; k < 3; ++k) {
			std::cout << NodeType::sz[k] << ", ";
		}
		std::cout << NodeType::sz[3] << ") and expected (";
		for (int k = 0; k < 3; ++k) {
			std::cout << sz_exp[k] << ", ";
		}
		std::cout << sz_exp[3] << ")" << std::endl;
		assert(false);
	}
	if (sz_w[3] % numGroup_ != 0 || sz_w[3] <= 0) {
		std::cout << "Convolution filters not correct." << std::endl;
		assert(false);
	}
	if (sz_w[2] * numGroup_ != sz_phi[2]) {
		std::cout << "Convolution channels don't match." << std::endl;
		assert(false);
	}
}

template <class N>
void ConvFunction<N>::Evaluate(TreePostIter& cur, STATE){
	//implement input1 conv input2
	assert(cur.LSDN_NUMBER_OF_CHILDREN() > (hasBias_ ? 2ul : 1ul));
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz_w = c_b_ptr->GetSizePtr();
	ValueType* val_w = c_b_ptr->GetValuePtr();
	SizeType numDim_w = c_b_ptr->GetNumDim();
	//SizeType numEl_w  = c_b_ptr->GetNumEl();
	assert(numDim_w == 4);
	//assume sz_w: h, w, channel, num_filters

	++c_b;
	c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	SizeType* sz_phi = c_b_ptr->GetSizePtr();
	ValueType* val_phi = c_b_ptr->GetValuePtr();
	SizeType numDim_phi = c_b_ptr->GetNumDim();
	//SizeType numEl_phi  = c_b_ptr->GetNumEl();
	assert(numDim_phi == 4);

	if (NodeType::sz == NULL && NodeType::value == NULL){
		NodeType::numDim = numDim_w;
		NodeType::sz = new SizeType[NodeType::numDim];

		NodeType::sz[0] = (sz_phi[0] + 2*padSize_ - sz_w[0]) / stride_ + 1;
		NodeType::sz[1] = (sz_phi[1] + 2*padSize_ - sz_w[1]) / stride_ + 1;
		NodeType::sz[2] = sz_w[3];
		NodeType::sz[3] = sz_phi[3];

		NodeType::ComputeNumEl();
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}
	SizeType valueSampleDim = NodeType::sz[0] * NodeType::sz[1] * NodeType::sz[2];

	if (patchMatrix_ == NULL) {
		sz_pm_ = new SizeType[2];
		sz_pm_[0] = NodeType::sz[0]*NodeType::sz[1];
		sz_pm_[1] = sz_w[0]*sz_w[1]*sz_phi[2];

		patchMatrix_ = NodeType::AllocValueMem(typename NodeType::GPUType(), sz_pm_[0] * sz_pm_[1]);
	}

	//dimensions for matrix multiplication
	SizeType dimM = NodeType::sz[0] * NodeType::sz[1];
	SizeType dimN = NodeType::sz[2] / numGroup_;
	SizeType dimK = sz_w[0] * sz_w[1] * sz_w[2];  //dimension of input w should be carefully handled beforehand

	SizeType sampleDim = sz_phi[0] * sz_phi[1] * sz_phi[2];  //the dimension of a sample
	SizeType val_w_offset = dimK * dimN;
	SizeType p_offset     = dimM * dimK;
	SizeType v_offset     = dimM * dimN;

	//iterate over num_input
	for(SizeType k = 0; k < sz_phi[3]; ++k) {
		Im2Patches(typename NodeType::GPUType(), val_phi + k*sampleDim, sz_phi[0], sz_phi[1], sz_phi[2], sz_w[0], sz_w[1]);

		for(SizeType g = 0; g < numGroup_; ++g) {
			MultiplyMatMat(typename NodeType::GPUType(), patchMatrix_ + p_offset*g, val_w + val_w_offset*g, NodeType::value + valueSampleDim*k + v_offset*g, dimM, dimN, dimK, CblasNoTrans, CblasNoTrans, 1.0, 0.0);
		}
	}

	++c_b;
	if (hasBias_) {
		assert(c_b != cur.end());
		c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
		ValueType* val_next = c_b_ptr->GetValuePtr();
		SizeType* sz_next = c_b_ptr->GetSizePtr();
		SizeType numDim_next = c_b_ptr->GetNumDim();
		if (numDim_next == 1) {
			assert(sz_next[0]==NodeType::sz[2]);
			AdditionModuloOperand(typename NodeType::GPUType(), NodeType::value, NodeType::numEl, val_next, NodeType::sz[0] * NodeType::sz[1], sz_next[0]);
		} else if (numDim_next == 3) {
			assert(sz_next[0] == NodeType::sz[0] && sz_next[1] == NodeType::sz[1] && sz_next[2] == NodeType::sz[2]);
			AdditionModuloOperand(typename NodeType::GPUType(), NodeType::value, NodeType::numEl, val_next, 1, c_b_ptr->GetNumEl());
		}
		++c_b;
	}

	for (typename NodeType::TreeSiblIter ptr_e = cur.end(); c_b != ptr_e; ++c_b) {
		c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
		ValueType* val_next = c_b_ptr->GetValuePtr();
		SizeType numEl_next = c_b_ptr->GetNumEl();
		assert(NodeType::numEl == numEl_next);
		VectorAdd(typename NodeType::GPUType(), numEl_next, ValueType(1.0), val_next, NodeType::value);
	}

	if (hasRELU_) {
		ReluFunction<N>::PerformReluForward(typename NodeType::GPUType(), NodeType::value, NodeType::numEl, NodeType::value);
	}
}

template <class N>
void ConvFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() > (hasBias_ ? 2ul : 1ul));
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz_w = c_b_ptr->GetSizePtr();
	SizeType numEl_w = c_b_ptr->GetNumEl();
	ValueType* val_w = c_b_ptr->GetValuePtr();
	//SizeType numDim_w = c_b_ptr->GetNumDim();
	ValueType** output_w = c_b_ptr->GetDiffGradientAndEmpMean();
#ifndef LSDN_USE_GRAPH
	NODETYPE type_w = c_b_ptr->IdentifyMe();
#endif
	assert(output_w != NULL);

	++c_b;
	c_b_ptr = LSDN_NODE_ACCESSOR(c_b);
	SizeType* sz_phi = c_b_ptr->GetSizePtr();
	SizeType numEl_phi = c_b_ptr->GetNumEl();
	ValueType* val_phi = c_b_ptr->GetValuePtr();
	//SizeType numDim_phi = c_b_ptr->GetNumDim();
	ValueType** output_phi = c_b_ptr->GetDiffGradientAndEmpMean();

	if (hasRELU_) {
		ReluFunction<N>::PerformReluBackwardNoAdd(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::value, NodeType::numEl, ComputeFunction<NodeType>::DiffGradNEmpMean);
	}

	//for first input
	if (*output_w == NULL) {
		*output_w = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_w);
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_w, 0, numEl_w);
	}
#ifndef LSDN_USE_GRAPH
	else if (type_w != NODE_PARAM) {
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_w, 0, numEl_w);
	}
#endif

	if (patchMatrix_ == NULL) {
		assert(false); //something wrong; patcmMatrix_ should have been allocated
	}
	if (patchMatrixDiff_ == NULL) {
		patchMatrixDiff_ = NodeType::AllocValueMem(typename NodeType::GPUType(), sz_pm_[0] * sz_pm_[1]);
	}

	//dimensions for matrix multiplication
	SizeType dimM = NodeType::sz[0] * NodeType::sz[1];
	SizeType dimN = NodeType::sz[2] / numGroup_;
	SizeType dimK = sz_w[0] * sz_w[1] * sz_w[2];  //dimension of input w should be carefully handled beforehand

	SizeType valueSampleDim = NodeType::sz[0] * NodeType::sz[1] * NodeType::sz[2];
	SizeType sampleDim = sz_phi[0] * sz_phi[1] * sz_phi[2];  //the dimension of a sample
	SizeType val_w_offset = dimK * dimN;
	SizeType p_offset     = dimM * dimK;
	SizeType v_offset     = dimM * dimN;

	if (output_phi != NULL) {
		if (*output_phi == NULL) {
			*output_phi = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl_phi);
			LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_phi, 0, numEl_phi);
		}
#ifndef LSDN_USE_GRAPH
		else if (c_b_ptr->IdentifyMe() != NODE_PARAM) {
			LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output_phi, 0, numEl_phi);
		}
#endif
	}

	//iterate over num_input
	for(SizeType k = 0; k < sz_phi[3]; ++k) {
		Im2Patches(typename NodeType::GPUType(), val_phi + sampleDim*k, sz_phi[0], sz_phi[1], sz_phi[2], sz_w[0], sz_w[1]);

		//diff w.r.t. first input
		for(SizeType g = 0; g < numGroup_; ++g) {
			MultiplyMatMat(typename NodeType::GPUType(), patchMatrix_ + p_offset*g, ComputeFunction<NodeType>::DiffGradNEmpMean + valueSampleDim*k + v_offset*g, *output_w + val_w_offset*g, dimK, dimN, dimM, CblasTrans, CblasNoTrans);
		}

		//diff w.r.t. second input
		//for second input
		if (output_phi != NULL) {
			//LSDNMemSet<ValueType>(typename NodeType::GPUType(), patchMatrixDiff_, 0, sz_pm_[0] * sz_pm_[1]);

			for(SizeType g = 0; g < numGroup_; ++g) {
				MultiplyMatMat(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean + valueSampleDim*k + v_offset*g, val_w + val_w_offset*g, patchMatrixDiff_+p_offset*g, dimM, dimK, dimN, CblasNoTrans, CblasTrans, ValueType(1.0), ValueType(0.0));
			}

			AccumPatchDiff2Im(typename NodeType::GPUType(), *output_phi + sampleDim*k, sz_phi[0], sz_phi[1], sz_phi[2], sz_w[0], sz_w[1]);
		}
	}

	++c_b;
	if (hasBias_) {
		assert(c_b != cur.end());
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
			if (numDim_next == 1) {
				assert(sz_next[0] == NodeType::sz[2]);

				BiasDerivativeSingleDim(typename NodeType::GPUType(), *output_next, ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::sz[0] * NodeType::sz[1], NodeType::sz[3], NodeType::sz[2], 
#ifdef LSDN_USE_GRAPH
					true
#else
					(c_b_ptr->IdentifyMe() == NODE_PARAM)
#endif
					);
			} else if (numDim_next == 3) {
				assert(sz_next[0] == NodeType::sz[0] && sz_next[1] == NodeType::sz[1] && sz_next[2] == NodeType::sz[2]);

				BiasDerivativeMultiDim(typename NodeType::GPUType(), *output_next, ComputeFunction<NodeType>::DiffGradNEmpMean, NodeType::sz[0] * NodeType::sz[1] * NodeType::sz[2], NodeType::sz[3], 
#ifdef LSDN_USE_GRAPH
					true
#else
					(c_b_ptr->IdentifyMe() == NODE_PARAM)
#endif
					);
			}
		}
		++c_b;
	}

	//diff w.r.t. remaining inputs
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
typename ConvFunction<N>::ValueType ConvFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template class ConvFunction<Node<double, int, false> >;
template class ConvFunction<Node<double, int, true> >;
template class ConvFunction<Node<float, int, false> >;
template class ConvFunction<Node<float, int, true> >;
