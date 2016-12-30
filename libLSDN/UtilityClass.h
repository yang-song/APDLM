//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __UTILITYCLASS_H__
#define __UTILITYCLASS_H__
#include <cstddef>
#include <vector>

template <typename V, typename S, bool G, bool P>
class UtilityClass {
	typedef V ValueType;
	typedef S SizeType;
public:
	UtilityClass();
	~UtilityClass();

	template <typename T>
	void AdjustComputationPointers(ValueType* basePtr, std::vector<T*>& vec, std::vector<bool>& isRootContainer, size_t* ValuesRootOffset);
	template <typename T>
	int UpdateCPUDataOffset(ValueType* basePtr, std::vector<T*>& vec, std::vector<bool>& isRootContainer, bool ignoreNonRoot);
	template <typename T>
	void AdjustDerivativePointers(ValueType* baseDataPtr, std::vector<T*>& vec, std::vector<bool>& isRootContainer, ValueType* baseCPUDerivativeRootPtr);
};

#endif