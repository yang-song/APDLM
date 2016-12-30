//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __LSDN_H__
#define __LSDN_H__
#include <random>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

class LSDN
{
private:
	static void CleanUp();
	LSDN();
	~LSDN();
	LSDN(LSDN const&);
	LSDN& operator=(LSDN const&);

	static LSDN* m_instance;
private:
	cublasHandle_t cublas_handle_;
	curandGenerator_t gen_GPU;
	std::mt19937 gen_CPU;
public:
	static LSDN& Instance();

	cublasHandle_t& cublas_handle();
	curandGenerator_t& curand_generator();
	std::mt19937& cpu_generator();
	void SetSeed(int seed);
};

#endif