#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <assert.h>
#include <string.h>
#include <float.h>
//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <algorithm>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "Function_Interpolate.h"

#include "LSDN_mathfunctions.h"

template <class N>
InterpolateFunction<N>::InterpolateFunction(const NodeParameters& params) : int_h_(params.int_h), int_w_(params.int_w) {}

template <class N>
InterpolateFunction<N>::~InterpolateFunction() {}

template <class N>
void InterpolateFunction<N>::Clear() {
	ComputeFunction<N>::Clear();
}

template <class N>
void InterpolateFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	assert(numDim1 == NodeType::numDim);
	assert(numDim1 > 1);

	SizeType h_out = (sz1[0]-1) * int_h_ + 1;
	SizeType w_out = (sz1[1]-1)*int_w_ + 1;
	assert(NodeType::sz[0] == h_out);
	assert(NodeType::sz[1] == w_out);

	for (SizeType k = 2; k < numDim1; ++k) {
		NodeType::sz[k] = sz1[k];
	}
	NodeType::ComputeNumEl();
}

template <class N>
void InterpolateFunction<N>::BilinearInterpolateForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	for (SizeType p = 0; p < NodeType::numEl; ++p) {
		SizeType h = p%h_out;
		SizeType w = (p / h_out) % w_out;
		SizeType offset = p / (h_out*w_out);

		SizeType h_in_topleft = h / int_h_;
		SizeType w_in_topleft = w / int_w_;

		SizeType h_insidebox = h%int_h_;
		SizeType w_insidebox = w%int_w_;

		ValueType* bot_ptr = bottom_pt + offset*h_bot*w_bot;
		if (h_insidebox == 0 && w_insidebox == 0) {
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft];
		} else if (h_insidebox == 0 && w_insidebox != 0) {//y = 0; x = w_insidebox/int_w_
			ValueType x = ValueType(w_insidebox) / ValueType(int_w_);
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x) + x*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft];
		} else if (h_insidebox != 0 && w_insidebox == 0) {//y = h_insidebox/int_h_; x = 0;
			ValueType y = ValueType(h_insidebox) / ValueType(int_h_);
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - y) + y*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1];
		} else {
			ValueType x = ValueType(w_insidebox) / ValueType(int_w_);
			ValueType y = ValueType(h_insidebox) / ValueType(int_h_);
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x)*(ValueType(1) - y) + x*(ValueType(1) - y)*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft] +
				y*(ValueType(1) - x)*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1] + x*y*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft + 1];
		}
	}
}

template <class N>
void InterpolateFunction<N>::BilinearInterpolateBackward(i2t<false>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	for (SizeType p = 0; p < numel_bot; ++p) {
		SizeType h = p%h_bot;
		SizeType w = (p / h_bot) % w_bot;
		SizeType offset = p / (h_bot*w_bot);

		SizeType h_top_coord = h*int_h_;
		SizeType w_top_coord = w*int_w_;

		SizeType hc_start = std::max(h_top_coord - int_h_ + 1, 0);
		SizeType hc_end = std::min(h_top_coord + int_h_, h_out);
		SizeType wc_start = std::max(w_top_coord - int_w_ + 1, 0);
		SizeType wc_end = std::min(w_top_coord + int_w_, w_out);

		ValueType* top_ptr = top_diff + offset*h_out*w_out;
		ValueType val = ValueType(0);
		for (SizeType wc = wc_start; wc < wc_end; ++wc) {
			ValueType x = ValueType(1.0) - fabs(ValueType(wc - w_top_coord)) / int_w_;
			for (SizeType hc = hc_start; hc < hc_end; ++hc) {
				ValueType y = ValueType(1.0) - fabs(ValueType(hc - h_top_coord)) / int_h_;
				val += x*y*top_ptr[wc*h_out + hc];
			}
		}

		bot_diff[p] += val;
	}
}

template <class N>
void InterpolateFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	assert(numDim1 > 1);

	SizeType h_out = (sz1[0]-1)*int_h_ + 1;
	SizeType w_out = (sz1[1]-1)*int_w_ + 1;

	if (NodeType::sz == NULL && NodeType::value == NULL){
		NodeType::sz = new SizeType[numDim1];
		NodeType::sz[0] = h_out;
		NodeType::sz[1] = w_out;
		for (SizeType k = 2; k < numDim1; ++k) {
			NodeType::sz[k] = sz1[k];
		}

		NodeType::numDim = numDim1;
		NodeType::ComputeNumEl();
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}
	assert(NodeType::numEl != 0);

	ValueType* bottom_pt = val1;
	ValueType* top_pt = NodeType::value;

	BilinearInterpolateForward(typename NodeType::GPUType(), bottom_pt, sz1[0], sz1[1], top_pt, h_out, w_out);
}

template <class N>
void InterpolateFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	//ValueType* val1 = c_b_ptr->GetValuePtr();
	//SizeType numDim1 = c_b_ptr->GetNumDim();
	SizeType numEl1 = c_b_ptr->GetNumEl();
	ValueType** output = c_b_ptr->GetDiffGradientAndEmpMean();
	assert(output != NULL);

	assert(NodeType::numEl != 0);
	if (*output == NULL) {
		*output = NodeType::AllocValueMem(typename NodeType::GPUType(), numEl1);
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output, 0, numEl1);
	}
#ifndef LSDN_USE_GRAPH
	else if (c_b_ptr->IdentifyMe() != NODE_PARAM) {
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), *output, 0, numEl1);
	}
#endif

	SizeType h_out = NodeType::sz[0];
	SizeType w_out = NodeType::sz[1];

	ValueType* bot_diff_pt = *output;
	ValueType* top_diff_pt = ComputeFunction<NodeType>::DiffGradNEmpMean;

	BilinearInterpolateBackward(typename NodeType::GPUType(), bot_diff_pt, sz1[0], sz1[1], numEl1, top_diff_pt, h_out, w_out);

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template <class N>
typename InterpolateFunction<N>::ValueType InterpolateFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}


template class InterpolateFunction<Node<double, int, false> >;
template class InterpolateFunction<Node<double, int, true> >;
template class InterpolateFunction<Node<float, int, false> >;
template class InterpolateFunction<Node<float, int, true> >;



