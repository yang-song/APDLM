//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "LSDN.h"

LSDN* LSDN::m_instance = 0;

LSDN& LSDN::Instance() {
	if (m_instance == 0) {
		m_instance = new LSDN();
	}
	return *m_instance;
}

LSDN::LSDN() {
	cublas_handle_ = NULL;
	gen_GPU = NULL;
	atexit(&CleanUp);
	if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
		std::cout << "GPU not available because of cublas." << std::endl;
	}
	if (curandCreateGenerator(&gen_GPU, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS || curandSetPseudoRandomGeneratorSeed(gen_GPU, 1) != CURAND_STATUS_SUCCESS) {
		std::cout << "GPU not available because of random number generator." << std::endl;
	}
	gen_CPU.seed(1);
}

LSDN::~LSDN() {
	if (cublas_handle_) {
		cublasDestroy(cublas_handle_);
		cublas_handle_ = NULL;
	}
	if (gen_GPU) {
		curandDestroyGenerator(gen_GPU);
		gen_GPU = NULL;
	}
}

void LSDN::CleanUp() {
	if (m_instance) {
		m_instance->~LSDN();
		delete m_instance;
		m_instance = 0;
	}
}

void LSDN::SetSeed(int seed) {
	gen_CPU.seed(seed);
	if (curandSetPseudoRandomGeneratorSeed(gen_GPU, 1) != CURAND_STATUS_SUCCESS) {
		std::cout << "Failed to set seed of GPU random number generator." << std::endl;
	}
}

cublasHandle_t& LSDN::cublas_handle() {
	return Instance().cublas_handle_;
}

curandGenerator_t& LSDN::curand_generator() {
	return Instance().gen_GPU;
}

std::mt19937& LSDN::cpu_generator() {
	return Instance().gen_CPU;
}
