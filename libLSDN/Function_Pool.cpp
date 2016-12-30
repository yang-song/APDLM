//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <assert.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "Function_Pool.h"

#include "LSDN_mathfunctions.h"

template <class N>
PoolFunction<N>::PoolFunction(const NodeParameters& params) : kernel_h_(params.kh), kernel_w_(params.kw), padSize_(params.p), stride_(params.s), SubsampleH_(params.SubsampleH), SubsampleW_(params.SubsampleW), pool_method_(params.t) {}

template <class N>
PoolFunction<N>::~PoolFunction() {}

template <class N>
void PoolFunction<N>::Clear() {
	ComputeFunction<N>::Clear();
}

template <class N>
void PoolFunction<N>::AdjustDimension(TreePostIter& cur) {
	assert(false);//currently broken
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	assert(numDim1 == 4);
	assert(NodeType::numDim == 4);

	SizeType h_out = (sz1[0] - kernel_h_ + 2 * padSize_) / stride_ + 1;
	SizeType w_out = (sz1[1] - kernel_w_ + 2 * padSize_) / stride_ + 1;
	assert(NodeType::sz[0] == h_out);
	assert(NodeType::sz[1] == w_out);
	assert(NodeType::sz[2] == sz1[2]);

	if (NodeType::sz[3] == -1) {
		NodeType::sz[3] = sz1[3];
		NodeType::ComputeNumEl();
	}
}

template <class N>
void PoolFunction<N>::AvgPoolBackward(i2t<false>,ValueType* bot_diff_pt, SizeType h_bot, SizeType w_bot, ValueType* top_diff_pt, SizeType h_out, SizeType w_out) {
	for(SizeType k = 0; k < NodeType::sz[3]; ++k) {
		for(SizeType c = 0; c < NodeType::sz[2]; ++c) {
			for(SizeType pw = 0; pw < w_out; ++pw) {
				SizeType w_start = pw * stride_ - padSize_;
				SizeType w_end   = std::min(w_start + kernel_w_, w_bot + padSize_);

				for(SizeType ph = 0; ph < h_out; ++ph) {
					SizeType h_start   = ph * stride_ - padSize_;
					SizeType h_end     = std::min(h_start + kernel_h_, h_bot + padSize_);
					SizeType pool_size = (h_end - h_start) * (w_end - w_start);

					h_start = std::max(h_start, 0);
					w_start = std::max(w_start, 0);
					h_end   = std::min(h_end, h_bot);
					w_end   = std::min(w_end, w_bot);

					for(SizeType w = w_start; w < w_end; ++w) {
						for(SizeType h = h_start; h < h_end; ++h) {
							bot_diff_pt[w*h_bot+h] += top_diff_pt[pw*h_out+ph] / pool_size;
						}
					}
				}
			}

			bot_diff_pt += h_bot * w_bot;
			top_diff_pt += h_out * w_out;
		}
	}
}

template <class N>
void PoolFunction<N>::MaxPoolBackward(i2t<false>, ValueType* bot_data_pt, ValueType* bot_diff_pt, SizeType h_bot, SizeType w_bot,
		ValueType* top_data_pt, ValueType* top_diff_pt, SizeType h_out, SizeType w_out) {
	for(SizeType k = 0; k < NodeType::sz[3]; ++k) {
		for(SizeType c = 0; c < NodeType::sz[2]; ++c) {
			for(SizeType pw = 0; pw < w_out; ++pw) {
				SizeType w_start = pw * stride_ - padSize_;
				SizeType w_end   = std::min(w_start + kernel_w_*SubsampleW_, w_bot);

				if (w_start < 0) {
					w_start = w_start + ((-w_start - 1) / SubsampleW_ + 1)*SubsampleW_;
				}

				for(SizeType ph = 0; ph < h_out; ++ph) {
					SizeType h_start = ph * stride_ - padSize_;
					SizeType h_end   = std::min(h_start + kernel_h_*SubsampleH_, h_bot);

					if (h_start<0) {
						h_start = h_start + ((-h_start - 1) / SubsampleH_ + 1)*SubsampleH_;
					}

					if (h_start<h_end && w_start<w_end && h_start<h_bot && w_start<w_bot) {
					//if (h_end>0 && w_end>0 && h_start < h_bot && w_start < w_bot) {
					//	h_start = std::max(h_start, 0);
					//	w_start = std::max(w_start, 0);

						for (SizeType w = w_start; w < w_end; w += SubsampleW_) {
							for (SizeType h = h_start; h < h_end; h += SubsampleH_) {
								//should check how many bot_data = top_data, and divide the bot_diff by it
								//we assume only one bot_data = top_data for speed
								bot_diff_pt[w*h_bot + h] +=
									top_diff_pt[pw*h_out + ph] *
									(top_data_pt[pw*h_out + ph] == bot_data_pt[w*h_bot + h]);
							}
						}
					}
				}
			}

			bot_data_pt += h_bot * w_bot;
			bot_diff_pt += h_bot * w_bot;
			top_data_pt += h_out * w_out;
			top_diff_pt += h_out * w_out;
		}
	}
}

template <class N>
void PoolFunction<N>::AvgPoolForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	for(SizeType k = 0; k < NodeType::sz[3]; ++k) {
		for(SizeType c = 0; c < NodeType::sz[2]; ++c) {
			for(SizeType pw = 0; pw < w_out; ++pw) {
				SizeType w_start = pw * stride_ - padSize_;
				SizeType w_end   = std::min(w_start + kernel_w_, w_bot + padSize_);

				for(SizeType ph = 0; ph < h_out; ++ph) {
					SizeType h_start   = ph * stride_ - padSize_;
					SizeType h_end     = std::min(h_start + kernel_h_, h_bot + padSize_);
					SizeType pool_size = (h_end - h_start) * (w_end - w_start);

					h_start = std::max(h_start, 0);
					w_start = std::max(w_start, 0);
					h_end   = std::min(h_end, h_bot);
					w_end   = std::min(w_end, w_bot);

					for(SizeType w = w_start; w < w_end; ++w) {
						for(SizeType h = h_start; h < h_end; ++h) {
							top_pt[pw * h_out + ph] += bottom_pt[w * h_bot + h];
						}
					}
					top_pt[pw * h_out + ph] /= pool_size;
				}
			}

			bottom_pt += w_bot * h_bot;
			top_pt    += w_out * h_out;
		}
	}
}

template <class N>
void PoolFunction<N>::MaxPoolForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	std::fill(NodeType::value, NodeType::value + NodeType::numEl, ValueType(-FLT_MAX));

	for(SizeType k = 0; k < NodeType::sz[3]; ++k) {
		for(SizeType c = 0; c < NodeType::sz[2]; ++c) {
			for(SizeType pw = 0; pw < w_out; ++pw) {
				SizeType w_start = pw * stride_ - padSize_;
				SizeType w_end   = std::min(w_start + kernel_w_*SubsampleW_, w_bot);

				if (w_start < 0) {
					w_start = w_start + ((-w_start-1)/SubsampleW_ + 1)*SubsampleW_;
				}

				for(SizeType ph = 0; ph < h_out; ++ph) {
					SizeType h_start = ph * stride_ - padSize_;
					SizeType h_end   = std::min(h_start + kernel_h_*SubsampleH_, h_bot);

					if (h_start<0) {
						h_start = h_start + ((-h_start-1)/SubsampleH_+1)*SubsampleH_;
					}
					//h_start = std::max(h_start, 0);
					//w_start = std::max(w_start, 0);

					if (h_start<h_end && w_start<w_end && h_start<h_bot && w_start<w_bot) {
						for (SizeType w = w_start; w < w_end; w += SubsampleW_) {
							for (SizeType h = h_start; h < h_end; h += SubsampleH_) {
								top_pt[pw * h_out + ph] = std::max(top_pt[pw * h_out + ph], bottom_pt[w * h_bot + h]);
							}
						}
					} else {
						top_pt[pw*h_out + ph] = ValueType(0);
					}
				}
			}

			bottom_pt += w_bot * h_bot;
			top_pt    += w_out * h_out;
		}
	}

}

template <class N>
void PoolFunction<N>::Evaluate(TreePostIter& cur, STATE) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
	SizeType numDim1 = c_b_ptr->GetNumDim();
	//SizeType numEl1 = c_b_ptr->GetNumEl();
	assert(numDim1 == 4);

	//SizeType h_out = (sz1[0] - kernel_h_*SubsampleH_ + 2 * padSize_ + ((SubsampleH_>1) ? 1 : 0)) / stride_ + 1;
	//SizeType w_out = (sz1[1] - kernel_w_*SubsampleW_ + 2 * padSize_ + ((SubsampleW_>1) ? 1 : 0)) / stride_ + 1;
	SizeType h_out = (sz1[0] + 2 * padSize_ - ((kernel_h_-1)*SubsampleH_ + 1)) / stride_ + 1;
	SizeType w_out = (sz1[1] + 2 * padSize_ - ((kernel_w_-1)*SubsampleW_ + 1)) / stride_ + 1;

	if (NodeType::sz == NULL && NodeType::value == NULL){
		NodeType::sz = new SizeType[numDim1];
		NodeType::sz[0] = h_out;
		NodeType::sz[1] = w_out;
		NodeType::sz[2] = sz1[2];
		NodeType::sz[3] = sz1[3];

		NodeType::numDim = numDim1;
		NodeType::ComputeNumEl();
		NodeType::value = NodeType::AllocValueMem(typename NodeType::GPUType(), NodeType::numEl);
	}
	assert(NodeType::numEl != 0);

	ValueType* bottom_pt = val1;
	ValueType* top_pt   = NodeType::value;

	switch (pool_method_){
	case MAX_POOLING: //max pooling
		//assert(padSize_==0); //why caffe only implemented padding for average pooling?
		MaxPoolForward(typename NodeType::GPUType(), bottom_pt, sz1[0], sz1[1], top_pt, h_out, w_out);
		break;
	case AVG_POOLING: //average pooling
		assert(SubsampleH_ == 1 && SubsampleW_ == 1);
		LSDNMemSet<ValueType>(typename NodeType::GPUType(), NodeType::value, 0, NodeType::numEl);
		AvgPoolForward(typename NodeType::GPUType(), bottom_pt, sz1[0], sz1[1], top_pt, h_out, w_out);
		break;
	default:
		assert(false);
		break;
	}//end switch

}

template <class N>
void PoolFunction<N>::Gradient(TreePostIter& cur) {
	assert(cur.LSDN_NUMBER_OF_CHILDREN() == 1);
	typename NodeType::TreeSiblIter c_b = cur.begin();
	NodeType* c_b_ptr = LSDN_NODE_ACCESSOR(c_b);

	SizeType* sz1 = c_b_ptr->GetSizePtr();
	ValueType* val1 = c_b_ptr->GetValuePtr();
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

	SizeType h_out = NodeType::sz[0]; //(sz1[0] - kernel_h_ + 2*padSize_ ) / stride_ + 1;
	SizeType w_out = NodeType::sz[1]; //(sz1[1] - kernel_w_ + 2*padSize_ ) / stride_ + 1;

	ValueType* bot_data_pt  = val1;
	ValueType* bot_diff_pt  = *output;
	ValueType* top_data_pt  = NodeType::value;
	ValueType* top_diff_pt  = ComputeFunction<NodeType>::DiffGradNEmpMean;

	switch (pool_method_){
	case MAX_POOLING: //max pooling
		MaxPoolBackward(typename NodeType::GPUType(), bot_data_pt, bot_diff_pt, sz1[0], sz1[1],
			top_data_pt, top_diff_pt, h_out, w_out);
		break;
	case AVG_POOLING: //average pooling
		assert(SubsampleH_ == 1 && SubsampleW_ == 1);
		AvgPoolBackward(typename NodeType::GPUType(), bot_diff_pt, sz1[0], sz1[1],
			top_diff_pt, h_out, w_out);
		break;
	default:
		assert(false);
		break;
	}//end switch

#ifdef LSDN_USE_GRAPH
	LSDNMemSet<ValueType>(typename NodeType::GPUType(), ComputeFunction<NodeType>::DiffGradNEmpMean, 0, NodeType::numEl);
#endif
}

template <class N>
typename PoolFunction<N>::ValueType PoolFunction<N>::GetValue(size_t ix, int AccessOffset, int Stride) {
	return NodeType::value[ix*Stride + AccessOffset];
}


template class PoolFunction<Node<double, int, false> >;
template class PoolFunction<Node<double, int, true> >;
template class PoolFunction<Node<float, int, false> >;
template class PoolFunction<Node<float, int, true> >;



