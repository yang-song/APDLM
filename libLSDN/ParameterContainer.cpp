//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <iostream>
#include "ParameterContainer.h"

#include "cuda_runtime.h"

#include "../LSDN_CudaCommon.h"

#ifdef WITH_MPI
#include "mpi.h"
#endif

#define DEF_CUDA_FREE(x) \
if ((x) != NULL) { \
	cudaFree((x)); \
	(x) = NULL; \
}

template <typename V, typename S, bool G, bool P>
ParameterContainer<V, S, G, P>::ParameterContainer() : GPUParameterValues(NULL), GPUParameterDerivative(NULL), ParameterDiffHistory(NULL), GPUParameterValuesRootOffset(0) {}

template <typename V, typename S, bool G, bool P>
ParameterContainer<V, S, G, P>::~ParameterContainer() {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::Clear() {
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		(*it)->DeletedByContainer();
		(*it)->Clear();
		delete *it;
	}

	DEF_CUDA_FREE(GPUParameterValues)
	DEF_CUDA_FREE(GPUParameterDerivative)

	if (ParameterDiffHistory != NULL) {
		DeAllocValueMem(typename NodeType::GPUType(), ParameterDiffHistory);
		ParameterDiffHistory = NULL;
	}
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::DeAllocValueMem(i2t<true>, ValueType* ptr) {
	cudaFree(ptr);
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::DeAllocValueMem(i2t<false>, ValueType* ptr) {
	delete[] ptr;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ReduceStepSize() {
	for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		(*it)->ReduceStepSize();
	}
}

template <typename V, typename S, bool G, bool P>
typename ParameterContainer<V, S, G, P>::ParaType* ParameterContainer<V, S, G, P>::AddParameter(SizeType* sz, SizeType numDim, const typename ParaType::NodeParameters& params, bool isRoot, int paramID) {
	ParaType* retVal = new ParaType(params);
	retVal->SetValueSize(NULL, sz, numDim);
	ParameterIDToContainerPosition.insert(std::pair<int, int>(paramID, int(paramClasses.size())));
	paramClasses.push_back(retVal);
	ParamClassesRoot.push_back(isRoot);
	return retVal;
}

template <typename V, typename S, bool G, bool P>
int ParameterContainer<V, S, G, P>::CreateCPUMemoryForParameters() {
	size_t numData = 0;
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		numData += (*it)->GetNumEl();
	}
	CPUParameterValues.resize(numData);
    this->UpdateCPUDataOffset(&CPUParameterValues[0], paramClasses, ParamClassesRoot, false);
	return 0;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::PrepareComputation(STATE purpose) {
	ValueType* baseParameterPtr = &CPUParameterValues[0];
	CreateGPUParameterData(typename NodeType::GPUType(), baseParameterPtr);
    this->AdjustComputationPointers(baseParameterPtr, paramClasses, ParamClassesRoot, &GPUParameterValuesRootOffset);

	if (purpose == TRAIN) {
		CreateCPUDerivativeMemoryForParameters(typename NodeType::GPUType());
		ValueType* baseParameterDerivativePtr = NULL;
		ValueType* ParameterRootOffset = NULL;
		if (CPUParameterDerivative.size() > 0) {//mod-ify: no modification required
			baseParameterDerivativePtr = &CPUParameterDerivative[0];
			ParameterRootOffset = baseParameterDerivativePtr + ComputeRootFunctionOffset(typename NodeType::GPUType(), GPUParameterValuesRootOffset);
		}
		CreateGPUDerivativeMemoryForParameters(typename NodeType::GPUType(), baseParameterDerivativePtr);
        this->AdjustDerivativePointers(baseParameterDerivativePtr, paramClasses, ParamClassesRoot, ParameterRootOffset);//mod-ify: make sure root pointers fit in case of MPI; adjusted ComputeRootFunctionOffset(i2t<true>, size_t val)

		CreateHistoryData(typename NodeType::GPUType());
		AdjustHistoryPointers();
	}
}

template <typename V, typename S, bool G, bool P>
#ifdef WITH_MPI
size_t ParameterContainer<V, S, G, P>::ComputeRootFunctionOffset(i2t<true>, size_t val) {
	return val;
#else
size_t ParameterContainer<V, S, G, P>::ComputeRootFunctionOffset(i2t<true>, size_t) {
	return 0;
#endif
}

template <typename V, typename S, bool G, bool P>
size_t ParameterContainer<V, S, G, P>::ComputeRootFunctionOffset(i2t<false>, size_t val) {
	return val;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::AdjustHistoryPointers() {
	size_t numData = 0;
	std::vector<bool>::iterator isRoot = ParamClassesRoot.begin();
	for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it, ++isRoot) {
		if (!*isRoot) {
			ValueType** output = (*it)->GetDiffHist();
			assert(output != NULL && *output == NULL);
			*output = ParameterDiffHistory + numData;
			numData += (*it)->GetNumEl();
		}
	}
	isRoot = ParamClassesRoot.begin();
	for (typename std::vector<ParaType*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it, ++isRoot) {
		if (*isRoot) {
			ValueType** output = (*it)->GetDiffHist();
			assert(output != NULL && *output == NULL);
			*output = ParameterDiffHistory + numData;
			numData += (*it)->GetNumEl();
		}
	}
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUParameterData(i2t<true>, ValueType*& baseDataPtr) {
	assert(GPUParameterValues == NULL);
	cudaMalloc(&GPUParameterValues, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemcpy(GPUParameterValues, &CPUParameterValues[0], CPUParameterValues.size()*sizeof(ValueType), cudaMemcpyHostToDevice);
	check_cuda_errors(__FILE__, __LINE__);
	baseDataPtr = GPUParameterValues;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUParameterData(i2t<false>, ValueType*&) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateHistoryData(i2t<true>) {
	assert(ParameterDiffHistory == NULL);
	cudaMalloc(&ParameterDiffHistory, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemset(ParameterDiffHistory, 0, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateHistoryData(i2t<false>) {
	assert(ParameterDiffHistory == NULL);
	ParameterDiffHistory = new ValueType[CPUParameterValues.size()];
	std::fill(ParameterDiffHistory, ParameterDiffHistory + CPUParameterValues.size(), ValueType(0.0));
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUDerivativeMemoryForParameters(i2t<true>, ValueType*& baseDataPtr) {
	assert(GPUParameterDerivative == NULL);
	cudaMalloc(&GPUParameterDerivative, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	cudaMemset(GPUParameterDerivative, 0, CPUParameterValues.size()*sizeof(ValueType));
	check_cuda_errors(__FILE__, __LINE__);
	baseDataPtr = GPUParameterDerivative;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateGPUDerivativeMemoryForParameters(i2t<false>, ValueType*&) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateCPUDerivativeMemoryForParameters(i2t<false>) {
	CPUParameterDerivative.resize(CPUParameterValues.size(), ValueType(0.0));
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CreateCPUDerivativeMemoryForParameters(i2t<true>) {//mod-ify: generate complete array in case of MPI (not only root values)
	size_t numEl = 0;
	std::vector<bool>::iterator isRoot = ParamClassesRoot.begin();
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it, ++isRoot) {
#ifdef WITH_MPI
		numEl += (*it)->GetNumEl();
#else
		if (*isRoot) {
			numEl += (*it)->GetNumEl();
		}
#endif
	}
	CPUParameterDerivative.resize(numEl, ValueType(0.0));
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyRootParameterValues(i2t<true>) {
	if (CPUParameterValues.size()>GPUParameterValuesRootOffset) {
		cudaMemcpy(&CPUParameterValues[GPUParameterValuesRootOffset], GPUParameterValues + GPUParameterValuesRootOffset, (CPUParameterValues.size() - GPUParameterValuesRootOffset)*sizeof(ValueType), cudaMemcpyDeviceToHost);
		check_cuda_errors(__FILE__, __LINE__);
	}
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyRootParameterValues(i2t<false>) {}

template <typename V, typename S, bool G, bool P>//mod-ify: copy to the right location in case of MPI
void ParameterContainer<V, S, G, P>::CopyRootParameterDerivatives(i2t<true>) {//assumes size of CPUParameterDerivative to equal number of root elements
#ifdef WITH_MPI
	if (GPUParameterValuesRootOffset < CPUParameterValues.size()) {
		cudaMemcpy(GPUParameterDerivative + GPUParameterValuesRootOffset, &CPUParameterDerivative[0] + GPUParameterValuesRootOffset, (CPUParameterDerivative.size() - GPUParameterValuesRootOffset)*sizeof(ValueType), cudaMemcpyHostToDevice);
		check_cuda_errors(__FILE__, __LINE__);
	}
#else
	if (CPUParameterDerivative.size() > 0) {
		cudaMemcpy(GPUParameterDerivative + GPUParameterValuesRootOffset, &CPUParameterDerivative[0], CPUParameterDerivative.size()*sizeof(ValueType), cudaMemcpyHostToDevice);
		check_cuda_errors(__FILE__, __LINE__);
	}
#endif
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyRootParameterDerivatives(i2t<false>) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesFromGPU(i2t<false>) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesFromGPU(i2t<true>) {
#ifdef WITH_MPI
	cudaMemcpy(&CPUParameterDerivative[0], GPUParameterDerivative, CPUParameterDerivative.size()*sizeof(ValueType), cudaMemcpyDeviceToHost);
	check_cuda_errors(__FILE__, __LINE__);
#else
	assert(false);
#endif
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesToGPU(i2t<false>) {}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::CopyParameterDerivativesToGPU(i2t<true>) {
#ifdef WITH_MPI
	cudaMemcpy(GPUParameterDerivative, &CPUParameterDerivative[0], CPUParameterDerivative.size()*sizeof(ValueType), cudaMemcpyHostToDevice);
	check_cuda_errors(__FILE__, __LINE__);
#else
	assert(false);
#endif
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::Update(int ClusterSize) {//mod-ify: copy derivative from GPU to CPU; MPIAllreduce; copy back to GPU
#ifdef WITH_MPI
	if (ClusterSize > 1) {
		CopyParameterDerivativesFromGPU(typename NodeType::GPUType());
		ParameterBuffer.resize(CPUParameterDerivative.size());
		std::copy(CPUParameterDerivative.begin(), CPUParameterDerivative.end(), ParameterBuffer.begin());
		//std::vector<ValueType> tmp_buffer(CPUParameterDerivative);
		std::cout << "Transferring " << CPUParameterDerivative.size() << std::endl;
		if (sizeof(ValueType) == 4) {
			MPI::COMM_WORLD.Allreduce(&ParameterBuffer[0], &CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::FLOAT, MPI::SUM);
		} else if (sizeof(ValueType) == 8) {
			MPI::COMM_WORLD.Allreduce(&ParameterBuffer[0], &CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::DOUBLE, MPI::SUM);
		} else {
			assert(false);
		}
		/*if (sizeof(ValueType) == 4) {
			MPI::COMM_WORLD.Reduce(&ParameterBuffer[0], &CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::FLOAT, MPI::SUM, 0);
			MPI::COMM_WORLD.Bcast(&CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::FLOAT, 0);
		} else if (sizeof(ValueType) == 8) {
			MPI::COMM_WORLD.Reduce(&ParameterBuffer[0], &CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::DOUBLE, MPI::SUM, 0);
			MPI::COMM_WORLD.Bcast(&CPUParameterDerivative[0], CPUParameterDerivative.size(), MPI::DOUBLE, 0);
		} else {
			assert(false);
		}*/
		CopyParameterDerivativesToGPU(typename NodeType::GPUType());
	}
#endif
	if (ClusterSize){}
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		(*it)->UpdateParameters();
	}
}

template <typename V, typename S, bool G, bool P>
V ParameterContainer<V, S, G, P>::GetRegularization() {
	ValueType reg = 0;
	for (typename std::vector<Parameters<NodeType>*>::iterator it = paramClasses.begin(), it_e = paramClasses.end(); it != it_e; ++it) {
		reg += (*it)->GetRegularization();
	}
	return reg;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetGradient(i2t<true>) {
	if (GPUParameterDerivative != NULL) {
		cudaMemset(GPUParameterDerivative, 0, sizeof(ValueType)*GPUParameterValuesRootOffset);//not required to clear root part on GPU since it's computed on the CPU and then copied to GPU; CPU memory is cleared in ResetCPURootParameterDerivative(i2t<true>)
		//cudaMemset(GPUParameterDerivative, 0, sizeof(ValueType)*CPUParameterValues.size());
		check_cuda_errors(__FILE__, __LINE__);
	}
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetGradient(i2t<false>) {
	std::fill(CPUParameterDerivative.begin(), CPUParameterDerivative.begin() + GPUParameterValuesRootOffset, ValueType(0.0));//remaining part will be cleared by ResetCPURootParameterDerivative(i2t<false>)
	//std::fill(CPUParameterDerivative.begin(), CPUParameterDerivative.end(), ValueType(0.0));
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetCPURootParameterDerivative(i2t<true>) {//mod-ify: in case of MPI we only clear the root part and not everything
#ifdef WITH_MPI
	std::fill(CPUParameterDerivative.begin() + GPUParameterValuesRootOffset, CPUParameterDerivative.end(), ValueType(0.0));
#else
	std::fill(CPUParameterDerivative.begin(), CPUParameterDerivative.end(), ValueType(0.0));
#endif
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::ResetCPURootParameterDerivative(i2t<false>) {
	std::fill(CPUParameterDerivative.begin() + GPUParameterValuesRootOffset, CPUParameterDerivative.end(), ValueType(0.0));
}

template <typename V, typename S, bool G, bool P>
typename std::vector<typename ParameterContainer<V, S, G, P>::ParaType*>* ParameterContainer<V, S, G, P>::GetParamClasses() {
	return &paramClasses;
}

template <typename V, typename S, bool G, bool P>
typename std::vector<typename ParameterContainer<V, S, G, P>::ValueType>* ParameterContainer<V, S, G, P>::GetWeights(i2t<false>) {
	return &CPUParameterValues;
}

template <typename V, typename S, bool G, bool P>
typename std::vector<typename ParameterContainer<V, S, G, P>::ValueType>* ParameterContainer<V, S, G, P>::GetWeights(i2t<true>) {
	cudaMemcpy(&CPUParameterValues[0], GPUParameterValues, CPUParameterValues.size()*sizeof(ValueType), cudaMemcpyDeviceToHost);
	check_cuda_errors(__FILE__, __LINE__);
	return &CPUParameterValues;
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::SetWeights(i2t<false>, std::vector<ValueType>* weights) {
	assert(weights->size() == CPUParameterValues.size());
	std::copy(weights->begin(), weights->end(), CPUParameterValues.begin());
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::SetWeights(i2t<true>, std::vector<ValueType>* weights) {
	assert(weights->size() == CPUParameterValues.size());
	cudaMemcpy(GPUParameterValues, &((*weights)[0]), weights->size()*sizeof(ValueType), cudaMemcpyHostToDevice);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename V, typename S, bool G, bool P>
size_t ParameterContainer<V, S, G, P>::GetWeightDimension() {
	return CPUParameterValues.size();
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::GetDerivative(i2t<false>, std::vector<ValueType>& deriv) {
	deriv.assign(CPUParameterDerivative.begin(), CPUParameterDerivative.end());
}

template <typename V, typename S, bool G, bool P>
void ParameterContainer<V, S, G, P>::GetDerivative(i2t<true>, std::vector<ValueType>& deriv) {
	deriv.assign(CPUParameterValues.size(), ValueType(0.0));
	cudaMemcpy(&deriv[0], GPUParameterDerivative, deriv.size()*sizeof(ValueType), cudaMemcpyDeviceToHost);
	check_cuda_errors(__FILE__, __LINE__);
}

template <typename V, typename S, bool G, bool P>
typename ParameterContainer<V, S, G, P>::ParaType* ParameterContainer<V, S, G, P>::GetPtrFromID(int id) {
	std::map<int, int>::iterator iter = ParameterIDToContainerPosition.find(id);
	if (iter == ParameterIDToContainerPosition.end()) {
		return NULL;
	} else {
		return paramClasses[iter->second];
	}
}

template class ParameterContainer<double, int, false, false>;
template class ParameterContainer<double, int, true, false>;
template class ParameterContainer<float, int, false, false>;
template class ParameterContainer<float, int, true, false>;
