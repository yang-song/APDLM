//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __LSDN_MATHFUNCTIONS_H__
#define __LSDN_MATHFUNCTIONS_H__

#include "../LSDN_Common.h"

#include "cblas.h"

void MultiplyMatVec(i2t<false>, double* A, double* x, double* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, double alpha = 1.0, double beta = 1.0);
void MultiplyMatVec(i2t<false>, float* A, float* x, float* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, float alpha = 1.0, float beta = 1.0);
void MultiplyMatMat(i2t<false>, double* A, double* B, double* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, double alpha = 1.0, double beta = 1.0);
void MultiplyMatMat(i2t<false>, float* A, float* B, float* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, float alpha = 1.0, float beta = 1.0);
void ScaleVecOrMat(i2t<false>, double* A, int dim, double scale);
void ScaleVecOrMat(i2t<false>, float* A, int dim, float scale);
void VectorInnerProduct(i2t<false>, int dim, float* x, float* y, float* result);
void VectorInnerProduct(i2t<false>, int dim, double* x, double* y, double* result);
void VectorAdd(i2t<false>, int dim, double alpha, double* x, double* y);
void VectorAdd(i2t<false>, int dim, float alpha, float* x, float* y);

void MultiplyMatVec(i2t<true>, double* A, double* x, double* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, double alpha = 1.0, double beta = 1.0);
void MultiplyMatVec(i2t<true>, float* A, float* x, float* res, int dim1, int dim2, const CBLAS_TRANSPOSE trans, float alpha = 1.0, float beta = 1.0);
void MultiplyMatMat(i2t<true>, double* A, double* B, double* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, double alpha = 1.0, double beta = 1.0);
void MultiplyMatMat(i2t<true>, float* A, float* B, float* res, int dim1, int dim2, int dim3, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, float alpha = 1.0, float beta = 1.0);
void ScaleVecOrMat(i2t<true>, double* A, int dim, double scale);
void ScaleVecOrMat(i2t<true>, float* A, int dim, float scale);
void VectorInnerProduct(i2t<true>, int dim, float* x, float* y, float* result);
void VectorInnerProduct(i2t<true>, int dim, double* x, double* y, double* result);
void VectorAdd(i2t<true>, int dim, double alpha, double* x, double* y);
void VectorAdd(i2t<true>, int dim, float alpha, float* x, float* y);

template <class T>
void ElementwiseExp(i2t<false>, T* data_in, T* data_out, size_t numEl);

template <class T>
void ElementwiseExp(i2t<true>, T* data_in, T* data_out, size_t numEl);

template <class T>
void NormLimitByCol(i2t<true>, int row, int col, T norm_limit, T* in);

template <class T>
void NormLimitByCol(i2t<false>, int row, int col, T norm_limit, T* in);

template <class T>
void LSDNMemSet(i2t<true>, T* dst, T val, size_t numEl);

template <class T>
void LSDNMemSet(i2t<false>, T* dst, T val, size_t numEl);

void LSDNMemCpy(i2t<true>, void* dst, void* src, size_t bytes);

void LSDNMemCpy(i2t<false>, void* dst, void* src, size_t bytes);

#endif