//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include <float.h>

#include "Function_Lrn.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"
#include "LSDN_mathfunctions.h"

template <typename T>
__global__ void kernel_LrnComputeDiff(const int num, T* bot_data, T* top_data, T* top_diff, T* scale,
		int numRow, int numCol, int numChan, int lrn_size, T negative_beta, T diffCoeff, T* bot_diff) {
	CUDA_KERNEL_LOOP(index, num) {
		// find out the local offset
		int h = index % numRow;
		int w = (index / numRow) % numCol;
		int n = index / numRow / numCol;
		int offset = (n * numChan * numCol + w) * numRow + h;
		int step   = numRow * numCol;

		bot_data += offset;
		top_data += offset;
		scale    += offset;
		top_diff += offset;
		bot_diff += offset;

		int head = 0;
		//for first lrn definition
		//int pre_pad = (lrn_size - 1) / 2;
		//int post_pad = lrn_size - pre_pad - 1;
		//for second lrn definition
		int post_pad = lrn_size / 2;
		T accum_ratio = 0;

		// accumulate values
		while (head < post_pad) {
			accum_ratio += top_diff[head * step] * top_data[head * step] /
					scale[head * step];
			++head;
		}
		// until we reach size, nothing needs to be subtracted
		while (head < lrn_size) {
			accum_ratio += top_diff[head * step] * top_data[head * step] /
					scale[head * step];
			bot_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step] *
					pow(scale[(head - post_pad) * step], negative_beta) - diffCoeff *
					bot_data[(head - post_pad) * step] * accum_ratio;
			++head;
		}
		// both add and subtract
		while (head < numChan) {
			accum_ratio += top_diff[head * step] * top_data[head * step] /
					scale[head * step];
			accum_ratio -= top_diff[(head - lrn_size) * step] *
					top_data[(head - lrn_size) * step] / scale[(head - lrn_size) * step];
			bot_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step] *
					pow(scale[(head - post_pad) * step], negative_beta) - diffCoeff *
					bot_data[(head - post_pad) * step] * accum_ratio;
			++head;
		}
		// subtract only
		while (head < numChan + post_pad) {
			accum_ratio -= top_diff[(head - lrn_size) * step] *
					top_data[(head - lrn_size) * step] / scale[(head - lrn_size) * step];
			bot_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step] *
					pow(scale[(head - post_pad) * step], negative_beta) - diffCoeff *
					bot_data[(head - post_pad) * step] * accum_ratio;
			++head;
		}
	}
}

template <class N>
void LrnFunction<N>::LrnAcrossChannelBackward(i2t<true>, ValueType* bot_data, ValueType* bot_diff) {
	int num_kernels = NodeType::sz[0] * NodeType::sz[1] * NodeType::sz[3];
	kernel_LrnComputeDiff<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), bot_data,
			NodeType::value, ComputeFunction<NodeType>::DiffGradNEmpMean, scale_data_,
				int(NodeType::sz[0]), int(NodeType::sz[1]), int(NodeType::sz[2]), lrn_size_, -beta_, ValueType(2.0 * alpha_ * beta_ / lrn_size_), bot_diff);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_LrnComputeOutput(const int num, T* in, T* scale, const T negative_beta, T* out) {
	CUDA_KERNEL_LOOP(index, num) {
		out[index] = in[index] * pow(scale[index], negative_beta);
	}
}

template <typename T>
__global__ void kernel_LrnComputeScaleData(const int num, T* in, int numRow, int numCol, int numChan,
		int lrn_size, T alpha_over_size, T* scale) {
	CUDA_KERNEL_LOOP(index, num) {
		int h = index % numRow;
		int w = (index / numRow) % numCol;
		int n = index / numRow / numCol;
		int offset = (n * numChan * numCol + w) * numRow + h;
		int step   = numRow * numCol;

		in += offset;
		scale += offset;

		int head = 0;
		//for first lrn definition
		//int pre_pad = (lrn_size - 1) / 2;
		//for second lrn definition
		int pre_pad = lrn_size / 2;

		int post_pad = lrn_size - pre_pad - 1;
		T accum_scale = 0;

		// fill the scale at [h, w, :, n];
	    // accumulate values
	    while (head < post_pad) {
	      accum_scale += in[head * step] * in[head * step];
	      ++head;
	    }
	    // until we reach size, nothing needs to be subtracted
	    // i.e., compute for first few channels
	    while (head < lrn_size) {
	      accum_scale += in[head * step] * in[head * step];
	      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
	      ++head;
	    }
	    // both add and subtract for middle channels
	    while (head < numChan) {
	      accum_scale += in[head * step] * in[head * step];
	      accum_scale -= in[(head - lrn_size) * step] * in[(head - lrn_size) * step];
	      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
	      ++head;
	    }
	    // subtract only for end channels
	    while (head < numChan + post_pad) {
	      accum_scale -= in[(head - lrn_size) * step] * in[(head - lrn_size) * step];
	      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
	      ++head;
	    }
	}
}

template <class N>
void LrnFunction<N>::LrnAcrossChannelForward(i2t<true>, ValueType* in) {
	//1. compute the scale_data_
	//use one kernel for one pixel (in output) and go through channels
	int num_kernels = NodeType::sz[0] * NodeType::sz[1] * NodeType::sz[3];
	kernel_LrnComputeScaleData<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), in,
			int(NodeType::sz[0]), int(NodeType::sz[1]), int(NodeType::sz[2]), lrn_size_, ValueType(alpha_ / lrn_size_), scale_data_);
	check_cuda_errors(__FILE__, __LINE__);

	//2. compute outputs
	num_kernels *= NodeType::sz[2];
	kernel_LrnComputeOutput<ValueType><<<LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS>>>(int(num_kernels), in,
			scale_data_, -beta_, NodeType::value);
	check_cuda_errors(__FILE__, __LINE__);
}


template class LrnFunction<Node<double, int, false> >;
template class LrnFunction<Node<double, int, true> >;
template class LrnFunction<Node<float, int, false> >;
template class LrnFunction<Node<float, int, true> >;



