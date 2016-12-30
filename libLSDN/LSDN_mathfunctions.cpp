//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <algorithm>
#include <string.h>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "LSDN.h"
#include "LSDN_mathfunctions.h"
#include "../LSDN_CudaCommon.h"

void MultiplyMatVec(i2t<false>, double* A, double* x, double* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, double alpha, double beta) {
	cblas_dgemv(CblasColMajor, trans, dim1, dim2, alpha, A, dim1, x, 1, beta, res, 1);
}
void MultiplyMatVec(i2t<false>, float* A, float* x, float* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, float alpha, float beta) {
	cblas_sgemv(CblasColMajor, trans, dim1, dim2, alpha, A, dim1, x, 1, beta, res, 1);
}
void MultiplyMatMat(i2t<false>, double* A, double* B, double* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, double alpha, double beta) {
	int lda = (transA == CblasNoTrans) ? dim1 : dim3;
	int ldb = (transB == CblasNoTrans) ? dim3 : dim2;
	cblas_dgemm(CblasColMajor, transA, transB, dim1, dim2, dim3, alpha, A, lda, B, ldb, beta, res, dim1);
}
void MultiplyMatMat(i2t<false>, float* A, float* B, float* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, float alpha, float beta) {
	int lda = (transA == CblasNoTrans) ? dim1 : dim3;
	int ldb = (transB == CblasNoTrans) ? dim3 : dim2;
	cblas_sgemm(CblasColMajor, transA, transB, dim1, dim2, dim3, alpha, A, lda, B, ldb, beta, res, dim1);
}
void ScaleVecOrMat(i2t<false>, double* A, int dim, double scale){
	cblas_dscal(dim, scale, A, 1);
}
void ScaleVecOrMat(i2t<false>, float* A, int dim, float scale){
	cblas_sscal(dim, scale, A, 1);
}
void VectorInnerProduct(i2t<false>, int dim, float* x, float* y, float* result){
	*result = cblas_sdot(dim, x, 1, y, 1);
}
void VectorInnerProduct(i2t<false>, int dim, double* x, double* y, double* result){
	*result = cblas_ddot(dim, x, 1, y, 1);
}
void VectorAdd(i2t<false>, int dim, double alpha, double* x, double* y) {
	cblas_daxpy(dim, alpha, x, 1, y, 1);
}
void VectorAdd(i2t<false>, int dim, float alpha, float* x, float* y) {
	cblas_saxpy(dim, alpha, x, 1, y, 1);
}

void MultiplyMatVec(i2t<true>, double* A, double* x, double* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, double alpha, double beta) {
	cublasOperation_t op = (trans == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasStatus_t error = cublasDgemv(LSDN::Instance().cublas_handle(), op, dim1, dim2, &alpha, A, dim1, x, 1, &beta, res, 1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void MultiplyMatVec(i2t<true>, float* A, float* x, float* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, float alpha, float beta) {
	cublasOperation_t op = (trans == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasStatus_t error = cublasSgemv(LSDN::Instance().cublas_handle(), op, dim1, dim2, &alpha, A, dim1, x, 1, &beta, res, 1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void MultiplyMatMat(i2t<true>, double* A, double* B, double* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, double alpha, double beta) {
	int lda = (transA == CblasNoTrans) ? dim1 : dim3;
	int ldb = (transB == CblasNoTrans) ? dim3 : dim2;
	cublasOperation_t opA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t opB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasStatus_t error = cublasDgemm(LSDN::Instance().cublas_handle(), opA, opB, dim1, dim2, dim3, &alpha, A, lda, B, ldb, &beta, res, dim1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void MultiplyMatMat(i2t<true>, float* A, float* B, float* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, float alpha, float beta) {
	int lda = (transA == CblasNoTrans) ? dim1 : dim3;
	int ldb = (transB == CblasNoTrans) ? dim3 : dim2;
	cublasOperation_t opA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t opB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasStatus_t error = cublasSgemm(LSDN::Instance().cublas_handle(), opA, opB, dim1, dim2, dim3, &alpha, A, lda, B, ldb, &beta, res, dim1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void ScaleVecOrMat(i2t<true>, double* A, int dim, double scale){
	cublasStatus_t error = cublasDscal(LSDN::Instance().cublas_handle(), dim, &scale, A, 1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void ScaleVecOrMat(i2t<true>, float* A, int dim, float scale){
	cublasStatus_t error = cublasSscal(LSDN::Instance().cublas_handle(), dim, &scale, A, 1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void VectorInnerProduct(i2t<true>, int dim, float* x, float* y, float* result) {
	cublasStatus_t error = cublasSdot(LSDN::Instance().cublas_handle(), dim, x, 1, y, 1, result);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void VectorInnerProduct(i2t<true>, int dim, double* x, double* y, double* result){
	cublasStatus_t error = cublasDdot(LSDN::Instance().cublas_handle(), dim, x, 1, y, 1, result);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void VectorAdd(i2t<true>, int dim, double alpha, double* x, double* y) {
	cublasStatus_t error = cublasDaxpy(LSDN::Instance().cublas_handle(), dim, &alpha, x, 1, y, 1);
	check_cublas_errors(__FILE__, __LINE__, error);
}
void VectorAdd(i2t<true>, int dim, float alpha, float* x, float* y) {
	cublasStatus_t error = cublasSaxpy(LSDN::Instance().cublas_handle(), dim, &alpha, x, 1, y, 1);
	check_cublas_errors(__FILE__, __LINE__, error);
}

template <class T>
void ElementwiseExp(i2t<false>, T* data_in, T* data_out, size_t numEl) {
	std::transform(data_in, data_in + numEl, data_out, (T(*)(T))&std::exp);
}
template void ElementwiseExp<double>(i2t<false>, double*, double*, size_t);
template void ElementwiseExp<float>(i2t<false>, float*, float*, size_t);

template <class T>
void LSDNMemSet(i2t<false>, T* dst, T val, size_t numEl) {
	std::fill(dst, dst + numEl, val);
}
template void LSDNMemSet<double>(i2t<false>, double*, double, size_t);
template void LSDNMemSet<float>(i2t<false>, float*, float, size_t);

void LSDNMemCpy(i2t<false>, void* dst, void* src, size_t bytes) {
	memcpy(dst, src, bytes);
}

template <class T>
void NormLimitByCol(i2t<false>, int row, int col, T norm_limit, T* in)
{
	//T sum = T(0.0);
	for (int c = 0; c < col; ++c) {
		T s = 0;
		VectorInnerProduct(i2t<false>(), row, in + c*row, in + c*row, &s);

		s = std::sqrt(s);

		T scale = (s > norm_limit) ? norm_limit / s : 1;

		if (scale != 1) {
			ScaleVecOrMat(i2t<false>(), in + c*row, row, scale);
		}
		//sum += s;
	}
	//std::cout << "Avg. Norm: " << sum / col << std::endl;
}

template void NormLimitByCol<double>(i2t<false>, int, int, double, double*);
template void NormLimitByCol<float>(i2t<false>, int, int, float, float*);
