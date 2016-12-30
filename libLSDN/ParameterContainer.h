//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __PARAMETERCONTAINER_H__
#define __PARAMETERCONTAINER_H__
#include <map>
#include "Parameters.h"
#include "UtilityClass.h"

template <typename V, typename S, bool G, bool P>
class ParameterContainer : public UtilityClass<V, S, G, P> {
public:
	typedef V ValueType;
	typedef S SizeType;
	typedef Node<V, S, G> NodeType;
	typedef Parameters<NodeType> ParaType;
private:
	std::map<int, int> ParameterIDToContainerPosition;

	std::vector<ParaType*> paramClasses;
	std::vector<bool> ParamClassesRoot;

	std::vector<ValueType> CPUParameterValues;
	std::vector<ValueType> CPUParameterDerivative;
	std::vector<ValueType> ParameterBuffer;
	ValueType* GPUParameterValues;
	ValueType* GPUParameterDerivative;
	ValueType* ParameterDiffHistory;
	size_t GPUParameterValuesRootOffset;
public:
	ParameterContainer();
	~ParameterContainer();
	virtual void Clear();
	virtual void DeAllocValueMem(i2t<true>, ValueType* ptr);
	virtual void DeAllocValueMem(i2t<false>, ValueType* ptr);

	virtual void ReduceStepSize();

	virtual ParaType* AddParameter(SizeType* sz, SizeType numDim, const typename ParaType::NodeParameters& params, bool isRoot, int paramID);

	virtual int CreateCPUMemoryForParameters();

	virtual void PrepareComputation(STATE purpose);
	virtual size_t ComputeRootFunctionOffset(i2t<true>, size_t);
	virtual size_t ComputeRootFunctionOffset(i2t<false>, size_t);

	virtual void CreateGPUParameterData(i2t<true>, ValueType*& baseDataPtr);
	virtual void CreateGPUParameterData(i2t<false>, ValueType*& baseDataPtr);
	virtual void CreateHistoryData(i2t<false>);
	virtual void CreateHistoryData(i2t<true>);
	virtual void AdjustHistoryPointers();

	virtual void CreateCPUDerivativeMemoryForParameters(i2t<false>);
	virtual void CreateCPUDerivativeMemoryForParameters(i2t<true>);
	virtual void CreateGPUDerivativeMemoryForParameters(i2t<false>, ValueType*& baseDataPtr);
	virtual void CreateGPUDerivativeMemoryForParameters(i2t<true>, ValueType*& baseDataPtr);

	virtual void CopyRootParameterValues(i2t<false>);
	virtual void CopyRootParameterValues(i2t<true>);
	virtual void CopyRootParameterDerivatives(i2t<false>);
	virtual void CopyRootParameterDerivatives(i2t<true>);
	virtual void CopyParameterDerivativesFromGPU(i2t<false>);
	virtual void CopyParameterDerivativesFromGPU(i2t<true>);
	virtual void CopyParameterDerivativesToGPU(i2t<false>);
	virtual void CopyParameterDerivativesToGPU(i2t<true>);

	virtual void Update(int ClusterSize);
	virtual ValueType GetRegularization();
	virtual void ResetGradient(i2t<true>);
	virtual void ResetGradient(i2t<false>);
	virtual void ResetCPURootParameterDerivative(i2t<true>);
	virtual void ResetCPURootParameterDerivative(i2t<false>);

	virtual std::vector<ParaType*>* GetParamClasses();
	virtual std::vector<ValueType>* GetWeights(i2t<true>);
	virtual std::vector<ValueType>* GetWeights(i2t<false>);
	virtual void SetWeights(i2t<true>, std::vector<ValueType>* weights);
	virtual void SetWeights(i2t<false>, std::vector<ValueType>* weights);
	virtual size_t GetWeightDimension();
	virtual void GetDerivative(i2t<false>, std::vector<ValueType>& deriv);
	virtual void GetDerivative(i2t<true>, std::vector<ValueType>& deriv);
	ParaType* GetPtrFromID(int id);
};

#endif