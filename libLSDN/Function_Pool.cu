//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <float.h>

#include <iostream>

#include "Function_Pool.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"
#include "LSDN_mathfunctions.h"

template <typename T>
__global__ void kernel_AvgPoolBackward(const int num, T* bot_diff, int numRow, int numCol, int numChan,
		int kNumRow, int kNumCol, int stride, int pad, T* top_diff, int h_out, int w_out) {
	CUDA_KERNEL_LOOP(index, num) {
		int h = index % numRow + pad;
		int w = (index / numRow) % numCol + pad;
		int c = (index / numRow / numCol) % numChan;
		int n = index / numRow / numCol / numChan;

		int ph_start = (h < kNumRow) ? 0 : (h - kNumRow) / stride + 1;
		int ph_end   = min(h / stride + 1, h_out);
		int pw_start = (w < kNumCol) ? 0 : (w - kNumCol) / stride + 1;
		int pw_end   = min(w / stride + 1, w_out);
		T grad = 0;

		top_diff += (n * numChan + c) * w_out * h_out;

		for (int pw = pw_start; pw < pw_end; ++pw) {
			for (int ph = ph_start; ph < ph_end; ++ph) {
				int h_start = ph * stride - pad;
				int w_start = pw * stride - pad;
				int h_end = min(h_start + kNumRow, numRow + pad);
				int w_end = min(w_start + kNumCol, numCol + pad);
				int pool_size = (h_end - h_start) * (w_end - w_start);

				grad += top_diff[pw * h_out + ph] / pool_size;
			}
		}

		bot_diff[index] = grad;
	}
}

template <typename T>
__global__ void kernel_MaxPoolBackward(const int num, T* bot_data, T* bot_diff, int numRow, int numCol, int numChan,
	int kNumRow, int kNumCol, int stride, int pad, T* top_data, T* top_diff, int h_out, int w_out, int subsample_h, int subsample_w) {
	CUDA_KERNEL_LOOP(index, num) {
		int h = index % numRow + pad;
		int w = (index / numRow) % numCol + pad;
		int c = (index / numRow / numCol) % numChan;
		int n = index / numRow / numCol / numChan;

		int ph_start = (h < kNumRow*subsample_h) ? h%subsample_h : (h - kNumRow*subsample_h) / stride + subsample_h;
		int ph_end   = min(h / stride + 1, h_out);
		int pw_start = (w < kNumCol*subsample_w) ? w%subsample_w : (w - kNumCol*subsample_w) / stride + subsample_w;
		int pw_end   = min( w / stride + 1, w_out);

		T grad = 0;
		//T bot = bot_data[((n * numChan + c) * numCol + w) * numRow + h];
		T bot = bot_data[index];
		top_data += (n * numChan + c) * w_out * h_out;
		top_diff += (n * numChan + c) * w_out * h_out;

		for (int pw = pw_start; pw < pw_end; pw+=subsample_w) {
			for (int ph = ph_start; ph < ph_end; ph+=subsample_h) {
				grad += top_diff[pw * h_out + ph] * (bot == top_data[pw * h_out + ph]);
			}
		}

		bot_diff[index] = grad;
	}
}

template <typename T>
__global__ void kernel_AvgPoolForward(const int num, T* bot, int numRow, int numCol, int numChan,
		int kNumRow, int kNumCol, int stride, int pad, T* top, int h_out, int w_out) {
 	CUDA_KERNEL_LOOP(index, num) {
 		int ph = index % h_out;
 		int pw = (index / h_out) % w_out;
 		int c  = (index / h_out / w_out) & numChan;
 		int n  = index / h_out / w_out / numChan;

 		int h_start = ph * stride - pad;
 		int w_start = pw * stride - pad;
 		int h_end = min(h_start + kNumRow, numRow + pad);
 		int w_end = min(w_start + kNumCol, numCol + pad);
 		int pool_size = (h_end - h_start) * (w_end - w_start);

 		h_start = max(h_start, 0);
 		w_start = max(w_start, 0);
 		h_end   = min(h_end, numRow);
 		w_end   = min(w_end, numCol);

 		T avg_val = 0;
 		bot += (n * numChan + c) * numRow * numCol;

 		for (int w = w_start; w < w_end; ++w) {
 			for (int h = h_start; h < h_end; ++h) {
 				avg_val += bot[w * numRow + h];
 			}
 		}

 		top[index] = avg_val / pool_size;
 	}
}

template <typename T>
__global__ void kernel_MaxPoolForward(const int num, T* bot, int numRow, int numCol, int numChan,
		int kNumRow, int kNumCol, int stride, int pad, T* top, int h_out, int w_out, int subsample_h, int subsample_w) {
 	CUDA_KERNEL_LOOP(index, num) {
 		int ph = index % h_out;
 		int pw = (index / h_out) % w_out;
 		int c  = (index / h_out / w_out) % numChan;
 		int n  = index / h_out / w_out / numChan;

 		int h_start = ph * stride - pad;
 		int h_end   = min(h_start + kNumRow*subsample_h, numRow);
 		int w_start = pw * stride - pad;
 		int w_end   = min(w_start + kNumCol*subsample_w, numCol);

		if (w_start < 0) {
			w_start = w_start + ((-w_start - 1) / subsample_w + 1)*subsample_w;
		}
		if (h_start<0) {
			h_start = h_start + ((-h_start - 1) / subsample_h + 1)*subsample_h;
		}

		if (h_start<h_end && w_start<w_end && h_start < numRow&&w_start < numCol) {
			T max_val = -FLT_MAX;
			bot += (n * numChan + c) * numCol * numRow;

			for (int w = w_start; w < w_end; w += subsample_w) {
				for (int h = h_start; h < h_end; h += subsample_h) {
					max_val = max(max_val, bot[w * numRow + h]);
				}
			}

			top[index] = max_val;
		} else {
			top[index] = T(0);
		}
 		
 	}
}

template <class N>
void PoolFunction<N>::AvgPoolBackward(i2t<true>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = h_bot * w_bot * NodeType::sz[2] * NodeType::sz[3];

	kernel_AvgPoolBackward<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), bot_diff,
			h_bot, w_bot, int(NodeType::sz[2]), int(kernel_h_), int(kernel_w_), int(stride_), int(padSize_), top_diff, int(h_out), int(w_out));
	check_cuda_errors(__FILE__, __LINE__);
}

template <class N>
void PoolFunction<N>::AvgPoolForward(i2t<true>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = NodeType::numEl;

	kernel_AvgPoolForward<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), bottom_pt,
			h_bot, w_bot, int(NodeType::sz[2]), int(kernel_h_), int(kernel_w_), int(stride_), int(padSize_), top_pt, int(h_out), int(w_out));
	check_cuda_errors(__FILE__, __LINE__);
}

template <class N>
void PoolFunction<N>::MaxPoolBackward(i2t<true>, ValueType* bot_data, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, ValueType* top_data, ValueType* top_diff, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = h_bot * w_bot * NodeType::sz[2] * NodeType::sz[3];

	kernel_MaxPoolBackward<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), bot_data, bot_diff,
		h_bot, w_bot, int(NodeType::sz[2]), int(kernel_h_), int(kernel_w_), int(stride_), int(padSize_), top_data, top_diff, int(h_out), int(w_out), int(SubsampleH_), int(SubsampleW_));
	check_cuda_errors(__FILE__, __LINE__);
}

template <class N>
void PoolFunction<N>::MaxPoolForward(i2t<true>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out) {
	SizeType num_kernels = NodeType::numEl;

	kernel_MaxPoolForward<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), bottom_pt,
			h_bot, w_bot, int(NodeType::sz[2]), int(kernel_h_), int(kernel_w_), int(stride_), int(padSize_), top_pt, int(h_out), int(w_out), int(SubsampleH_), int(SubsampleW_));
	check_cuda_errors(__FILE__, __LINE__);
}


template class PoolFunction<Node<double, int, false> >;
template class PoolFunction<Node<double, int, true> >;
template class PoolFunction<Node<float, int, false> >;
template class PoolFunction<Node<float, int, true> >;
