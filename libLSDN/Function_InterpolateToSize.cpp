//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <assert.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "Function_InterpolateToSize.h"

#include "LSDN_mathfunctions.h"

template <class N>
InterpolateToSizeFunction<N>::InterpolateToSizeFunction(const NodeParameters& params) : out_h_(params.out_h), out_w_(params.out_w) {}

template <class N>
InterpolateToSizeFunction<N>::~InterpolateToSizeFunction() {}

template <class N>
void InterpolateToSizeFunction<N>::Clear() {
	ComputeFunction<N>::Clear();
}

template <class N>
void InterpolateToSizeFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	assert(numDim1 == NodeType::numDim);
	assert(numDim1 > 1);

	SizeType h_out = out_h_;
	SizeType w_out = out_w_;
	assert(NodeType::sz[0] == h_out);
	assert(NodeType::sz[1] == w_out);

	for (SizeType k = 2; k < numDim1; ++k) {
		NodeType::sz[k] = sz1[k];
	}
	NodeType::ComputeNumEl();
}

template <class N>
void InterpolateToSizeFunction<N>::BilinearInterpolateForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	ValueType fraction_h = ValueType(h_bot - 1) / ValueType(h_out - 1);
	ValueType fraction_w = ValueType(w_bot - 1) / ValueType(w_out - 1);

	for (SizeType p = 0; p < NodeType::numEl; ++p) {
		SizeType h = p%h_out;
		SizeType w = (p / h_out) % w_out;
		SizeType offset = p / (h_out*w_out);

		ValueType factor_h = fraction_h*ValueType(h);
		ValueType factor_w = fraction_w*ValueType(w);

		SizeType h_in_topleft = SizeType(factor_h);
		SizeType w_in_topleft = SizeType(factor_w);

		ValueType y = factor_h - h_in_topleft;
		ValueType x = factor_w - w_in_topleft;

		ValueType* bot_ptr = bottom_pt + offset*h_bot*w_bot;
		if (h_in_topleft == h_bot-1 && w_in_topleft == w_bot-1) {
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft];
		} else if (h_in_topleft == h_bot - 1 && w_in_topleft != w_bot-1) {//y = 0; x = w_insidebox/int_w_
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x) + x*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft];
		} else if (h_in_topleft != h_bot - 1 && w_in_topleft == w_bot - 1) {//y = h_insidebox/int_h_; x = 0;
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - y) + y*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1];
		} else {
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x)*(ValueType(1) - y) + x*(ValueType(1) - y)*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft] +
				y*(ValueType(1) - x)*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1] + x*y*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft + 1];
		}
	}
}

template <class N>
void InterpolateToSizeFunction<N>::BilinearInterpolateBackward(i2t<false>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	ValueType fraction_h = ValueType(h_bot - 1) / ValueType(h_out - 1);
	ValueType fraction_w = ValueType(w_bot - 1) / ValueType(w_out - 1);

	for (SizeType p = 0; p < numel_bot; ++p) {
		SizeType h = p%h_bot;
		SizeType w = (p / h_bot) % w_bot;
		SizeType offset = p / (h_bot*w_bot);

		SizeType h_ceil_neg = SizeType(std::ceil((h - 1) / fraction_h));
		SizeType h_ceil_pos = SizeType(std::ceil((h + 1) / fraction_h));
		SizeType h_floor_neg = SizeType(std::floor((h - 1) / fraction_h));
		SizeType h_floor_pos = SizeType(std::floor((h + 1) / fraction_h));
		SizeType w_ceil_neg = SizeType(std::ceil((w - 1) / fraction_w));
		SizeType w_ceil_pos = SizeType(std::ceil((w + 1) / fraction_w));
		SizeType w_floor_neg = SizeType(std::floor((w - 1) / fraction_w));
		SizeType w_floor_pos = SizeType(std::floor((w + 1) / fraction_w));

		SizeType hc_start = std::max(h_ceil_neg + ((h_ceil_neg==h_floor_neg)?1:0), 0);
		SizeType hc_end = std::min(h_floor_pos - ((h_floor_pos==h_ceil_pos)?1:0)+1, h_out);
		SizeType wc_start = std::max(w_ceil_neg + ((w_ceil_neg == w_floor_neg) ? 1 : 0), 0);
		SizeType wc_end = std::min(w_floor_pos - ((w_floor_pos == w_ceil_pos) ? 1 : 0) + 1, w_out);
		ValueType hc_switch = h / fraction_h;
		ValueType wc_switch = w / fraction_w;

		ValueType* top_ptr = top_diff + offset*h_out*w_out;
		ValueType val = ValueType(0);
		for (SizeType wc = wc_start; wc < wc_end; ++wc) {
			
			ValueType x = fraction_w*ValueType(wc);
			x = x - SizeType(x);
			x = ((wc>=wc_switch) ? 1 - x : x);

			for (SizeType hc = hc_start; hc < hc_end; ++hc) {

				ValueType y = fraction_h*ValueType(hc);
				y = y - SizeType(y);
				y = ((hc>=hc_switch) ? 1 - y : y);

				val += x*y*top_ptr[wc*h_out + hc];
			}
		}

		bot_diff[p] += val;
	}
}

/*
template <class N>
void InterpolateToSizeFunction<N>::BilinearInterpolateForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	SizeType mod_h = (h_out-1)%(h_bot-1);
	SizeType floor_h = (h_out-1) / (h_bot-1);
	SizeType ceil_h = floor_h + ((mod_h != 0) ? 1 : 0);
	SizeType mod_w = (w_out-1)%(w_bot-1);
	SizeType floor_w = (w_out-1) / (w_bot-1);
	SizeType ceil_w = floor_w + ((mod_w != 0) ? 1 : 0);

	for (SizeType p = 0; p < NodeType::numEl; ++p) {
		SizeType h = p%h_out;
		SizeType w = (p / h_out) % w_out;
		SizeType offset = p / (h_out*w_out);

		SizeType factor_h = floor_h + ((h < mod_h*ceil_h) ? 1 : 0);
		SizeType factor_w = floor_w + ((w < mod_w*ceil_w) ? 1 : 0);

		SizeType h_in_topleft = ((h < mod_h*ceil_h) ? h / ceil_h : mod_h + (h - mod_h*ceil_h) / floor_h);
		SizeType w_in_topleft = ((w < mod_w*ceil_w) ? w / ceil_w : mod_w + (w - mod_w*ceil_w) / floor_w);

		SizeType h_insidebox = ((h < mod_h*ceil_h) ? h % ceil_h : (h - mod_h*ceil_h) % floor_h);
		SizeType w_insidebox = ((w < mod_w*ceil_w) ? w % ceil_w : (w - mod_w*ceil_w) % floor_w);

		ValueType* bot_ptr = bottom_pt + offset*h_bot*w_bot;
		if (h_insidebox == 0 && w_insidebox == 0) {
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft];
		} else if (h_insidebox == 0 && w_insidebox != 0) {//y = 0; x = w_insidebox/int_w_
			ValueType x = ValueType(w_insidebox) / ValueType(factor_w);
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x) + x*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft];
		} else if (h_insidebox != 0 && w_insidebox == 0) {//y = h_insidebox/int_h_; x = 0;
			ValueType y = ValueType(h_insidebox) / ValueType(factor_h);
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - y) + y*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1];
		} else {
			ValueType x = ValueType(w_insidebox) / ValueType(factor_w);
			ValueType y = ValueType(h_insidebox) / ValueType(factor_h);
			top_pt[p] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x)*(ValueType(1) - y) + x*(ValueType(1) - y)*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft] +
				y*(ValueType(1) - x)*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1] + x*y*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft + 1];
		}
	}
}

template <class N>
void InterpolateToSizeFunction<N>::BilinearInterpolateBackward(i2t<false>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	SizeType mod_h = (h_out - 1) % (h_bot - 1);
	SizeType floor_h = (h_out - 1) / (h_bot - 1);
	SizeType ceil_h = floor_h + ((mod_h != 0) ? 1 : 0);
	SizeType mod_w = (w_out - 1) % (w_bot - 1);
	SizeType floor_w = (w_out - 1) / (w_bot - 1);
	SizeType ceil_w = floor_w + ((mod_w != 0) ? 1 : 0);
	
	for (SizeType p = 0; p < numel_bot; ++p) {
		SizeType h = p%h_bot;
		SizeType w = (p / h_bot) % w_bot;
		SizeType offset = p / (h_bot*w_bot);

		SizeType h_top_coord = ((h < mod_h) ? h*ceil_h : mod_h*ceil_h + (h - mod_h)*floor_h);
		SizeType w_top_coord = ((w < mod_w) ? w*ceil_w : mod_w*ceil_w + (w - mod_w)*floor_w);

		SizeType beforeLength_h = (((h - 1) < mod_h) ? floor_h + 1 : floor_h);
		SizeType afterLength_h =  (((h    ) < mod_h) ? floor_h + 1 : floor_h);
		SizeType beforeLength_w = (((w - 1) < mod_w) ? floor_w + 1 : floor_w);
		SizeType afterLength_w =  (((w    ) < mod_w) ? floor_w + 1 : floor_w);

		SizeType hc_start = std::max(h_top_coord - beforeLength_h + 1, 0);
		SizeType hc_end = std::min(h_top_coord + afterLength_h, h_out);
		SizeType wc_start = std::max(w_top_coord - beforeLength_w + 1, 0);
		SizeType wc_end = std::min(w_top_coord + afterLength_w, w_out);

		ValueType* top_ptr = top_diff + offset*h_out*w_out;
		ValueType val = ValueType(0);
		for (SizeType wc = wc_start; wc < wc_end; ++wc) {
			ValueType x = ValueType(1.0) - fabs(ValueType(wc - w_top_coord)) / ((wc<w_top_coord)?beforeLength_w:afterLength_w);
			for (SizeType hc = hc_start; hc < hc_end; ++hc) {
				ValueType y = ValueType(1.0) - fabs(ValueType(hc - h_top_coord)) / ((hc<h_top_coord)?beforeLength_h:afterLength_h);
				val += x*y*top_ptr[wc*h_out + hc];
			}
		}

		bot_diff[p] += val;
	}
}
*/

template <class N>
void InterpolateToSizeFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	assert(numDim1 > 1);

	SizeType h_out = out_h_;
	SizeType w_out = out_w_;

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
void InterpolateToSizeFunction<N>::Gradient(TreePostIter& cur) {
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
typename InterpolateToSizeFunction<N>::ValueType InterpolateToSizeFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}


template class InterpolateToSizeFunction<Node<double, int, false> >;
template class InterpolateToSizeFunction<Node<double, int, true> >;
template class InterpolateToSizeFunction<Node<float, int, false> >;
template class InterpolateToSizeFunction<Node<float, int, true> >;



