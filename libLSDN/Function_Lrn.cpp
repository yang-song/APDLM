//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <math.h>

#include "Function_Lrn.h"
#include "LSDN_mathfunctions.h"

template <class N>
LrnFunction<N>::LrnFunction(const NodeParameters& params) : lrn_size_(params.l), alpha_(params.a), beta_(params.b), scale_data_(NULL) {}

template <class N>
LrnFunction<N>::~LrnFunction() {}

template <class N>
void LrnFunction<N>::Clear() {
	if (scale_data_ != NULL) {
		NodeType::DeAllocValueMem(typename NodeType::GPUType(), scale_data_);
		scale_data_ = NULL;
	}
	ComputeFunction<N>::Clear();
}

template <class N>
void LrnFunction<N>::LrnAcrossChannelBackward(i2t<false>, ValueType* val1, ValueType* output) {
	//// create buffers
	//for (v * f / scale) (v: diff of this layer, f: value of this layer, scale: precomputed during forward-pass)
	/*SizeType* sz_pad = new SizeType[3];
	sz_pad[0] = NodeType::sz[0];
	sz_pad[1] = NodeType::sz[1];
	sz_pad[2] = NodeType::sz[2] + lrn_size_ - 1;  //pad the data across channel axis
	SizeType numEl_pad = sz_pad[0] * sz_pad[1] * sz_pad[2];*/
	SizeType numEl_pad = NodeType::sz[0] * NodeType::sz[1] * (NodeType::sz[2] + lrn_size_ - 1);
	ValueType* val_pad = new ValueType[numEl_pad];
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), val_pad, 0, numEl_pad);

	//for (sum v * f / scale) (sum over neighboring channels)
	ValueType* val_accum = new ValueType[NodeType::sz[0] * NodeType::sz[1]];
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), val_accum, 0, NodeType::sz[0] * NodeType::sz[1]);

	//for ( (sum v * f / scale) * a )
	ValueType* val_accum_times_bot = new ValueType[NodeType::sz[0] * NodeType::sz[1]];

	SizeType padChannelStart = lrn_size_ - (lrn_size_+1) / 2;//caffe
	//SizeType padChannelStart = (lrn_size_ - 1) / 2;//nitish
	SizeType channelDim  = NodeType::sz[0]*NodeType::sz[1];
	SizeType sampleDim   = NodeType::sz[0]*NodeType::sz[1]*NodeType::sz[2];
	ValueType diffCoeff = ValueType(2.0) * alpha_ * beta_ / lrn_size_;

	for(SizeType k = 0; k < NodeType::numEl; ++k) {
		output[k] = ComputeFunction<NodeType>::DiffGradNEmpMean[k] * std::pow(scale_data_[k], -beta_);
	}

	//iterate over inputs
	for (SizeType k = 0; k < NodeType::sz[3]; ++k) {
		SizeType sampleOffset = sampleDim*k;

		//compute v*f/scale
		for (SizeType m = 0; m < sampleDim; ++m) {
			val_pad[channelDim*padChannelStart+m] = ComputeFunction<NodeType>::DiffGradNEmpMean[sampleOffset+m] * NodeType::value[sampleOffset+m] / scale_data_[sampleOffset+m];
		}

		LSDNMemSet<ValueType>(typename NodeType::GPUType(), val_accum, 0, NodeType::sz[0] * NodeType::sz[1]);

		//compute val_accum (sum before last channel of lrn_size) for 1st channel
		for (SizeType m = 0; m < lrn_size_-1; ++m) {
			VectorAdd(typename NodeType::GPUType(), channelDim, ValueType(1.0), val_pad+channelDim*m, val_accum);
		}

		//compute results for each channel
		for (SizeType m = 0; m < NodeType::sz[2]; ++m) {
			SizeType channelOffset = channelDim*m;

			//add new part for m-th channel
			VectorAdd(typename NodeType::GPUType(), channelDim, ValueType(1.0), val_pad+channelDim*(m+lrn_size_-1), val_accum);

			for(SizeType n = 0; n < channelDim; ++n){
				val_accum_times_bot[n] = val_accum[n] * val1[sampleOffset+channelOffset+n];
			}

			VectorAdd(typename NodeType::GPUType(), channelDim, -diffCoeff, val_accum_times_bot, output+sampleOffset+channelOffset);

			//delete old part for m-th channel
			VectorAdd(typename NodeType::GPUType(), channelDim, ValueType(-1.0), val_pad+channelOffset, val_accum);
		}
	}

	//delete[] sz_pad;
	delete[] val_pad;
	delete[] val_accum;
	delete[] val_accum_times_bot;
}

template <class N>
void LrnFunction<N>::LrnAcrossChannelForward(i2t<false>, ValueType* val1) {
	std::fill(scale_data_, scale_data_ + NodeType::numEl, ValueType(1.0));

	/*SizeType* sz_pad = new SizeType[3];
	sz_pad[0] = NodeType::sz[0];
	sz_pad[1] = NodeType::sz[1];
	sz_pad[2] = NodeType::sz[2] + lrn_size_ - 1;  //pad the data across channel axis
	SizeType numEl_pad = sz_pad[0] * sz_pad[1] * sz_pad[2];*/
	SizeType numEl_pad = NodeType::sz[0] * NodeType::sz[1] * (NodeType::sz[2] + lrn_size_ - 1);
	ValueType* val_pad = new ValueType[numEl_pad];

	LSDNMemSet<ValueType>(typename NodeType::GPUType(), val_pad, 0, numEl_pad);

	SizeType padChannelStart = (lrn_size_-1) / 2;//caffe
	//SizeType padChannelStart = lrn_size_ / 2;//nitish
	SizeType channelDim = NodeType::sz[0]*NodeType::sz[1];
	SizeType sampleDim  = NodeType::sz[0]*NodeType::sz[1]*NodeType::sz[2];
	ValueType alpha_over_size = alpha_ / lrn_size_;

	for(SizeType k = 0; k < NodeType::sz[3]; ++k) {
		SizeType sampleOffset = sampleDim*k;

		//square kth input, save to val_pad
		for(SizeType m = 0; m < sampleDim; ++m) {
			val_pad[channelDim*padChannelStart+m] = val1[sampleOffset+m] * val1[sampleOffset+m];
		}

		//get the scale_data for the first channel of kth sample
		//(accumulate the squared input of neighboring channels of 1st channel)
		for(SizeType m = 0; m < lrn_size_; ++m){
			VectorAdd(typename NodeType::GPUType(), channelDim, alpha_over_size, val_pad+channelDim*m, scale_data_+sampleOffset);
		}

		//get the scale_data for the remaining channel
		//some parts of scale_data_ can be reused
		for(SizeType m = 1; m < NodeType::sz[2]; ++m) {
			//copy from previous computed values
			LSDNMemCpy(typename NodeType::GPUType(), scale_data_+sampleOffset+channelDim*m, scale_data_+sampleOffset+channelDim*(m-1), sizeof(ValueType)*channelDim);

			//add one new channel
			VectorAdd(typename NodeType::GPUType(), channelDim, alpha_over_size, val_pad+channelDim*(m+lrn_size_-1), scale_data_+sampleOffset+channelDim*m);

			//delete one old channel
			VectorAdd(typename NodeType::GPUType(), channelDim, -alpha_over_size, val_pad+channelDim*(m-1), scale_data_+sampleOffset+channelDim*m);
		}
	}

	for(SizeType k = 0; k < NodeType::numEl; ++k) {
		NodeType::value[k] = val1[k] * std::pow(scale_data_[k], -beta_);
	}

	//delete[] sz_pad;
	delete[] val_pad;
}

template <class N>
void LrnFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	SizeType numEl1 = c_b_ptr->GetNumEl();
	assert(numDim1 == 4);
	assert(NodeType::numDim == 4);

	SizeType negPos = -1;
	for (SizeType k = 0; k < NodeType::numDim; ++k) {
		negPos = ((NodeType::sz[k] == -1) ? k : negPos);
		assert(NodeType::sz[k] == -1 || NodeType::sz[k] == sz1[k]);
	}
	if (negPos >= 0) {
		NodeType::sz[negPos] = sz1[negPos];
	}

	NodeType::numEl = numEl1;
}

template <class N>
void LrnFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	SizeType numEl1 = c_b_ptr->GetNumEl();
	assert(numDim1 == 4);

	if (NodeType::sz == NULL && NodeType::value == NULL){
		NodeType::sz = new SizeType[numDim1];
		memcpy((char*)NodeType::sz, (char*)sz1, numDim1*sizeof(SizeType));
		NodeType::numDim = numDim1;
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl1);
		NodeType::numEl = numEl1;
	}
	assert(NodeType::numEl != 0);

	if (scale_data_ == NULL) {
		scale_data_ = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl1);
	}

	LrnAcrossChannelForward(typename NodeType::GPUType(), val1);
}

template <class N>
void LrnFunction<N>::Gradient(TreePostIter& cur) {
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
	assert(false);//needs modification if used as node with multiple parents since the backward operation does not add
#endif

	assert(scale_data_ != NULL);

	LrnAcrossChannelBackward(typename NodeType::GPUType(), val1, *output);

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif

}

template <class N>
typename LrnFunction<N>::ValueType LrnFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}

template class LrnFunction<Node<double, int, false> >;
template class LrnFunction<Node<double, int, true> >;
template class LrnFunction<Node<float, int, false> >;
template class LrnFunction<Node<float, int, true> >;




