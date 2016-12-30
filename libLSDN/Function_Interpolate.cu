//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <float.h>

#include <iostream>

#include "Function_Interpolate.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"
#include "LSDN_mathfunctions.h"

template <typename T>
__global__ void kernel_BilinearInterpolateForward(const int num, T* bottom_pt, int h_bot, int w_bot, T* top_pt, int h_out, int w_out, int int_h_, int int_w_) {
	CUDA_KERNEL_LOOP(index, num) {
		int h = index%h_out;
		int w = (index / h_out) % w_out;
		int offset = index / (h_out*w_out);

		int h_in_topleft = h / int_h_;
		int w_in_topleft = w / int_w_;

		int h_insidebox = h%int_h_;
		int w_insidebox = w%int_w_;

		T* bot_ptr = bottom_pt + offset*h_bot*w_bot;
		if (h_insidebox == 0 && w_insidebox == 0) {
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft];
		} else if (h_insidebox == 0 && w_insidebox != 0) {//y = 0; x = w_insidebox/int_w_
			T x = T(w_insidebox) / T(int_w_);
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (T(1) - x) + x*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft];
		} else if (h_insidebox != 0 && w_insidebox == 0) {//y = h_insidebox/int_h_; x = 0;
			T y = T(h_insidebox) / T(int_h_);
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (T(1) - y) + y*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1];
		} else {
			T x = T(w_insidebox) / T(int_w_);
			T y = T(h_insidebox) / T(int_h_);
			top_pt[index] = bot_ptr[w_in_topleft*h_bot + h_in_topleft] * (T(1) - x)*(T(1) - y) + x*(T(1) - y)*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft] +
				y*(T(1) - x)*bot_ptr[w_in_topleft*h_bot + h_in_topleft + 1] + x*y*bot_ptr[(w_in_topleft + 1)*h_bot + h_in_topleft + 1];
		}
	}
}

template <typename T>
__global__ void kernel_BilinearInterpolateBackward(const int num, T* bot_diff, int h_bot, int w_bot, T* top_diff, int h_out, int w_out, int int_h_, int int_w_) {
	CUDA_KERNEL_LOOP(index, num) {
		int h = index%h_bot;
		int w = (index / h_bot) % w_bot;
		int offset = index / (h_bot*w_bot);

		int h_top_coord = h*int_h_;
		int w_top_coord = w*int_w_;

		int hc_start = max(h_top_coord - int_h_ + 1, 0);
		int hc_end = min(h_top_coord + int_h_, h_out);
		int wc_start = max(w_top_coord - int_w_ + 1, 0);
		int wc_end = min(w_top_coord + int_w_, w_out);

		T* top_ptr = top_diff + offset*h_out*w_out;
		T val = T(0);
		for (int wc = wc_start; wc < wc_end; ++wc) {
			T x = T(1.0) - fabs(T(wc - w_top_coord)) / int_w_;
			for (int hc = hc_start; hc < hc_end; ++hc) {
				T y = T(1.0) - fabs(T(hc - h_top_coord)) / int_h_;
				val += x*y*top_ptr[wc*h_out + hc];
			}
		}

		bot_diff[index] += val;
	}
}

template <class N>
void InterpolateFunction<N>::BilinearInterpolateForward(i2t<true>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = NodeType::numEl;

	kernel_BilinearInterpolateForward<ValueType> << <LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS >> >(int(num_kernels), bottom_pt, int(h_bot), int(w_bot), top_pt, int(h_out), int(w_out), int(int_h_), int(int_w_));
	check_cuda_errors(__FILE__, __LINE__);
}

template <class N>
void InterpolateFunction<N>::BilinearInterpolateBackward(i2t<true>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = numel_bot;

	kernel_BilinearInterpolateBackward<ValueType> << <LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS >> >(int(num_kernels), bot_diff, int(h_bot), int(w_bot), top_diff, int(h_out), int(w_out), int(int_h_), int(int_w_));
	check_cuda_errors(__FILE__, __LINE__);
}

template class InterpolateFunction<Node<double, int, false> >;
template class InterpolateFunction<Node<double, int, true> >;
template class InterpolateFunction<Node<float, int, false> >;
template class InterpolateFunction<Node<float, int, true> >;