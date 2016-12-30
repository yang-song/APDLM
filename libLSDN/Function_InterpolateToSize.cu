//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <float.h>

#include <iostream>

#include "Function_InterpolateToSize.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"
#include "LSDN_mathfunctions.h"

template <typename ValueType, typename SizeType>
__global__ void kernel_BilinearInterpolateToSizeForward(const int num, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	ValueType fraction_h = ValueType(h_bot - 1) / ValueType(h_out - 1);
	ValueType fraction_w = ValueType(w_bot - 1) / ValueType(w_out - 1);
	
	CUDA_KERNEL_LOOP(index, num) {
		SizeType h = index%h_out;
		SizeType w = (index / h_out) % w_out;
		SizeType offset = index / (h_out*w_out);

		ValueType factor_h = fraction_h*ValueType(h);
		ValueType factor_w = fraction_w*ValueType(w);

		SizeType h_in_topleft = SizeType(factor_h);
		SizeType w_in_topleft = SizeType(factor_w);

		ValueType y = factor_h - h_in_topleft;
		ValueType x = factor_w - w_in_topleft;

		ValueType* bot_ptr = bottom_pt + offset*h_bot*w_bot;
		if (h_in_topleft == h_bot - 1 && w_in_topleft == w_bot - 1) {
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft];
		} else if (h_in_topleft == h_bot - 1 && w_in_topleft != w_bot - 1) {//y = 0; x = w_insidebox/int_w_
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x) + x*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft];
		} else if (h_in_topleft != h_bot - 1 && w_in_topleft == w_bot - 1) {//y = h_insidebox/int_h_; x = 0;
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - y) + y*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1];
		} else {
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (ValueType(1) - x)*(ValueType(1) - y) + x*(ValueType(1) - y)*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft] +
				y*(ValueType(1) - x)*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1] + x*y*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft + 1];
		}
	}
}

template <typename ValueType, typename SizeType>
__global__ void kernel_BilinearInterpolateToSizeBackward(const int num, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	ValueType fraction_h = ValueType(h_bot - 1) / ValueType(h_out - 1);
	ValueType fraction_w = ValueType(w_bot - 1) / ValueType(w_out - 1);

	CUDA_KERNEL_LOOP(index, num) {
		SizeType h = index%h_bot;
		SizeType w = (index / h_bot) % w_bot;
		SizeType offset = index / (h_bot*w_bot);

		SizeType h_ceil_neg = SizeType(ceil((h - 1) / fraction_h));
		SizeType h_ceil_pos = SizeType(ceil((h + 1) / fraction_h));
		SizeType h_floor_neg = SizeType(floor((h - 1) / fraction_h));
		SizeType h_floor_pos = SizeType(floor((h + 1) / fraction_h));
		SizeType w_ceil_neg = SizeType(ceil((w - 1) / fraction_w));
		SizeType w_ceil_pos = SizeType(ceil((w + 1) / fraction_w));
		SizeType w_floor_neg = SizeType(floor((w - 1) / fraction_w));
		SizeType w_floor_pos = SizeType(floor((w + 1) / fraction_w));

		SizeType hc_start = max(h_ceil_neg + ((h_ceil_neg == h_floor_neg) ? 1 : 0), 0);
		SizeType hc_end = min(h_floor_pos - ((h_floor_pos == h_ceil_pos) ? 1 : 0) + 1, h_out);
		SizeType wc_start = max(w_ceil_neg + ((w_ceil_neg == w_floor_neg) ? 1 : 0), 0);
		SizeType wc_end = min(w_floor_pos - ((w_floor_pos == w_ceil_pos) ? 1 : 0) + 1, w_out);
		ValueType hc_switch = h / fraction_h;
		ValueType wc_switch = w / fraction_w;

		ValueType* top_ptr = top_diff + offset*h_out*w_out;
		ValueType val = ValueType(0);
		for (SizeType wc = wc_start; wc < wc_end; ++wc) {

			ValueType x = fraction_w*ValueType(wc);
			x = x - SizeType(x);
			x = ((wc >= wc_switch) ? 1 - x : x);

			for (SizeType hc = hc_start; hc < hc_end; ++hc) {

				ValueType y = fraction_h*ValueType(hc);
				y = y - SizeType(y);
				y = ((hc >= hc_switch) ? 1 - y : y);

				val += x*y*top_ptr[wc*h_out + hc];
			}
		}

		bot_diff[index] += val;
	}
}

template <class N>
void InterpolateToSizeFunction<N>::BilinearInterpolateForward(i2t<true>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = NodeType::numEl;

	kernel_BilinearInterpolateToSizeForward<ValueType, int> << <LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS >> >(int(num_kernels), bottom_pt, int(h_bot), int(w_bot), top_pt, int(h_out), int(w_out));
	check_cuda_errors(__FILE__, __LINE__);
}

template <class N>
void InterpolateToSizeFunction<N>::BilinearInterpolateBackward(i2t<true>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = numel_bot;
	
	kernel_BilinearInterpolateToSizeBackward<ValueType, int> << <LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS >> >(int(num_kernels), bot_diff, int(h_bot), int(w_bot), top_diff, int(h_out), int(w_out));
	check_cuda_errors(__FILE__, __LINE__);
}

template class InterpolateToSizeFunction<Node<double, int, false> >;
template class InterpolateToSizeFunction<Node<double, int, true> >;
template class InterpolateToSizeFunction<Node<float, int, false> >;
template class InterpolateToSizeFunction<Node<float, int, true> >;