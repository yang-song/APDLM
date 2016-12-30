//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include "Function_ConvSub.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"
#include "LSDN_mathfunctions.h"

template <typename T>
__global__ void kernel_AccumPatchDiff2ImSubsample(const int num, T* img, int numRow, int numCol, int numChan,
	int kNumRow, int kNumCol, int stride, int pad, const T* patchMatrixDiff, int h_out, int w_out, int subsample_h, int subsample_w) {
	CUDA_KERNEL_LOOP(index, num) {//loops over channels*numImageRows*numImageCols
		T res = 0;
		//coordinates in padded grid
		int h = index % numRow + pad;// +(kNumRow - 1)*(subsample_h / 2) - (subsample_h > 1)*pad;
		int w = (index / numRow) % numCol + pad;// +(kNumCol - 1)*(subsample_w / 2) - (subsample_w > 1)*pad;
		int c = index / (numRow * numCol);

		//where did pixel (index%numRow, (index/numRow)%numCol) have an influence in the patchmatrix?
		//the ranges in patchMatrixDiff that affect the current position (h,w) in img
		int h_out_start = (h < kNumRow*subsample_h) ? h%subsample_h : (h - kNumRow*subsample_h) / stride + subsample_h;
		int h_out_end = min(h / stride + 1, h_out);
		int w_out_start = (w < kNumCol*subsample_w) ? w%subsample_w : (w - kNumCol*subsample_w) / stride + subsample_w;
		int w_out_end = min(w / stride + 1, w_out);

		
		for (int w_c = w_out_start; w_c < w_out_end; w_c+=subsample_w) {
			for (int h_c = h_out_start; h_c < h_out_end; h_c+=subsample_h) {//height offset for grid in padded dimensions
				int c_out = c * kNumRow * kNumCol + ((w - w_c * stride)/subsample_w) * kNumRow + (h - h_c * stride)/subsample_h;
				res += patchMatrixDiff[(c_out * w_out + w_c) * h_out + h_c];
			}
		}
		//printf("%f ", patchMatrixDiff[0]);
		
		/*//equivalent implementation, fewer multiplications within for loops
		int offset = (c * kNumRow * kNumCol + w * kNumRow + h) * w_out * h_out;
		int coeff_h_out = (1 - stride * w_out * h_out);
		int coeff_w_out = (1 - stride * kNumRow * w_out) * h_out;

		for (int w_c = w_out_start; w_c < w_out_end; ++w_c) {
			for (int h_c = h_out_start; h_c < h_out_end; ++h_c) {
				res += patchMatrixDiff[offset + w_c*coeff_w_out + h_c*coeff_h_out];
			}
		}
		*/
		img[index] = res;

	}
}

template <class N>
void ConvSubFunction<N>::AccumPatchDiff2Im(i2t<true>, ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol) {
	//accumulated the gradients of extracted patches back to img

	//SizeType h_out = (numRow + 2 * padSize_ - kNumRow*SubsampleH_ + ((SubsampleH_>1) ? 1 : 0)) / stride_ + 1;
	//SizeType w_out = (numCol + 2 * padSize_ - kNumCol*SubsampleW_ + ((SubsampleW_>1) ? 1 : 0)) / stride_ + 1;
	SizeType h_out = (numRow + 2 * padSize_ - ((kNumRow - 1)*SubsampleH_ + 1)) / stride_ + 1;
	SizeType w_out = (numCol + 2 * padSize_ - ((kNumCol - 1)*SubsampleW_ + 1)) / stride_ + 1;
	SizeType num_kernels = numChan * numRow * numCol;

	// To avoid involving atomic operations, we launch one kernel per
	// input dimension, and then in the kernel add up the output dimensions.
	kernel_AccumPatchDiff2ImSubsample<ValueType> << <LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS >> >(int(num_kernels), img,
		int(numRow), int(numCol), int(numChan), int(kNumRow), int(kNumCol), int(stride_), int(padSize_),
		patchMatrixDiff_, int(h_out), int(w_out), int(SubsampleH_), int(SubsampleW_));

	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_Im2PatchesSubsample(const int num, const T* img, int numRow, int numCol,
	int kNumRow, int kNumCol, int stride, int pad, T* patchMatrix, int h_out, int w_out, int subsample_h, int subsample_w) {
	CUDA_KERNEL_LOOP(index, num) {//loops over channels*h_out*w_out
		int h_d = index % h_out;
		index /= h_out;
		int w_d = index % w_out;//width-destination
		int c_s = index / w_out;//channel-source
		int c_d = c_s * kNumRow * kNumCol;//channel-destination offset
		int h_s = h_d * stride - pad;// -(kNumRow - 1)*(subsample_h / 2) + (subsample_h > 1)*pad;
		int w_s = w_d * stride - pad;// -(kNumCol - 1)*(subsample_w / 2) + (subsample_w > 1)*pad;
		patchMatrix += (c_d * w_out + w_d) * h_out + h_d;
		img += (c_s * numCol + w_s) * numRow + h_s;

		for (int c = 0; c < kNumCol; ++c) {
			for (int r = 0; r < kNumRow; ++r) {
				int h = h_s + r*subsample_h;
				int w = w_s + c*subsample_w;

				*patchMatrix = (h >= 0 && h<numRow && w >= 0 && w<numCol) ? img[c*subsample_w*numRow + r*subsample_h] : 0;
				patchMatrix += w_out * h_out;
			}
		}

	}
}

template <class N>
void ConvSubFunction<N>::Im2Patches(i2t<true>, ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol) {
	//extract patches of size kNumRow-kNumCol-numChannel from img, and concatenate them into a matrix
	//each row stores the column vectorized of each patch

	//each kernel copies a single-channel grid (i.e., a kNumRow-kNumCol region)
	//SizeType h_out = (numRow + 2 * padSize_ - kNumRow*SubsampleH_ + ((SubsampleH_>1)?1:0)) / stride_ + 1;
	//SizeType w_out = (numCol + 2 * padSize_ - kNumCol*SubsampleW_ + ((SubsampleW_>1)?1:0)) / stride_ + 1;
	SizeType h_out = (numRow + 2 * padSize_ - ((kNumRow - 1)*SubsampleH_ + 1)) / stride_ + 1;
	SizeType w_out = (numCol + 2 * padSize_ - ((kNumCol - 1)*SubsampleW_ + 1)) / stride_ + 1;
	SizeType num_kernels = numChan * h_out * w_out;

	check_cuda_errors(__FILE__, __LINE__);
	kernel_Im2PatchesSubsample<ValueType> << <LSDN_GET_BLOCKS(num_kernels), LSDN_CUDA_NUM_THREADS >> >(int(num_kernels), img,
		int(numRow), int(numCol), int(kNumRow), int(kNumCol), int(stride_), int(padSize_),
		patchMatrix_, int(h_out), int(w_out), int(SubsampleH_), int(SubsampleW_));
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_AdditionModuloOperand(T* res, int numEl, const T* addend, int op_division, int op_modulo) {
	CUDA_KERNEL_LOOP(index, numEl) {
		int ix = (index / op_division) % op_modulo;
		res[index] += addend[ix];
	}
}

template <class N>
void ConvSubFunction<N>::AdditionModuloOperand(i2t<true>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo) {
	kernel_AdditionModuloOperand<ValueType> << <LSDN_GET_BLOCKS(numEl), LSDN_CUDA_NUM_THREADS >> >(res, numEl, addend, op_division, op_modulo);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_BiasDerivativeSingleDim(T* res, const T* input, const int patchSize, const int numSamples, const int numChannels, const int sampleSize, bool performAddition) {
	if (performAddition) {
		CUDA_KERNEL_LOOP(index, numChannels) {
			const T* ptr = input + patchSize*index;
			for (int k = 0; k < patchSize*numSamples; ++k) {
				int sample = k / patchSize;
				int offset = k % patchSize;
				res[index] += ptr[sampleSize*sample + offset];
			}
		}
	} else {
		CUDA_KERNEL_LOOP(index, numChannels) {
			const T* ptr = input + patchSize*index;
			res[index] = T(0);
			for (int k = 0; k < patchSize*numSamples; ++k) {
				int sample = k / patchSize;
				int offset = k % patchSize;
				res[index] += ptr[sampleSize*sample + offset];
			}
		}
	}
}

template <class N>
void ConvSubFunction<N>::BiasDerivativeSingleDim(i2t<true>, ValueType* res, ValueType* input, SizeType patchSize, SizeType numSamples, SizeType numChannels, bool performAddition) {
	kernel_BiasDerivativeSingleDim<ValueType> << <LSDN_GET_BLOCKS(numChannels), LSDN_CUDA_NUM_THREADS >> >(res, input, patchSize, numSamples, numChannels, patchSize*numChannels, performAddition);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T>
__global__ void kernel_BiasDerivativeMultiDim(T* res, const T* input, const int sampleSize, const int numSamples, bool performAddition) {
	if (performAddition) {
		CUDA_KERNEL_LOOP(index, sampleSize) {
			for (int k = 0; k < numSamples; ++k) {
				res[index] += input[k*sampleSize + index];
			}
		}
	} else {
		CUDA_KERNEL_LOOP(index, sampleSize) {
			res[index] = T(0);
			for (int k = 0; k < numSamples; ++k) {
				res[index] += input[k*sampleSize + index];
			}
		}
	}
}

template <class N>
void ConvSubFunction<N>::BiasDerivativeMultiDim(i2t<true>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition) {
	kernel_BiasDerivativeMultiDim<ValueType> << <LSDN_GET_BLOCKS(sampleSize), LSDN_CUDA_NUM_THREADS >> >(res, input, sampleSize, numSamples, performAddition);
	check_cuda_errors(__FILE__, __LINE__);
}

template class ConvSubFunction<Node<double, int, false> >;
template class ConvSubFunction<Node<double, int, true> >;
template class ConvSubFunction<Node<float, int, false> >;
template class ConvSubFunction<Node<float, int, true> >;
