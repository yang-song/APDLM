//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifdef _MSC_VER
#pragma warning( disable : 4661 )
#endif
#include "Function_Affine.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "../LSDN_CudaCommon.h"


template <typename T>
__global__ void kernel_AdditionModuloOperand(T* res, int numEl, const T* addend, int op_division, int op_modulo) {
	CUDA_KERNEL_LOOP(index, numEl) {
		int ix = (index / op_division) % op_modulo;
		res[index] += addend[ix];
	}
}

template <class N>
void AffineFunction<N>::AdditionModuloOperand(i2t<true>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo) {
	kernel_AdditionModuloOperand<ValueType><<<LSDN_GET_BLOCKS(numEl), LSDN_CUDA_NUM_THREADS>>>(res, numEl, addend, op_division, op_modulo);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename T, typename S>
__global__ void kernel_BiasDerivativeSingleDim(T* res, const T* input, S sampleSize, S numSamples, bool performAddition) {
	if (performAddition) {
		CUDA_KERNEL_LOOP(index, sampleSize) {
			const T* ptr = input + index;
			for (S k = 0; k < numSamples; ++k) {
				res[index] += ptr[k*sampleSize];
			}
		}
	} else {
		CUDA_KERNEL_LOOP(index, sampleSize) {
			const T* ptr = input + index;
			res[index] = T(0);
			for (S k = 0; k < numSamples; ++k) {
				res[index] += ptr[k*sampleSize];
			}
		}
	}
}

template <class N>
void AffineFunction<N>::BiasDerivativeSingleDim(i2t<true>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition) {
	kernel_BiasDerivativeSingleDim<ValueType, SizeType><<<LSDN_GET_BLOCKS(sampleSize), LSDN_CUDA_NUM_THREADS>>>(res, input, sampleSize, numSamples, performAddition);
	check_cuda_errors(__FILE__, __LINE__);
}

template class AffineFunction<Node<double, int, false> >;
template class AffineFunction<Node<double, int, true> >;
template class AffineFunction<Node<float, int, false> >;
template class AffineFunction<Node<float, int, true> >;