//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include <assert.h>
#include "UtilityClass.h"

#include "Node.h"
#include "Data.h"
#include "Parameters.h"
#include "ComputeFunction.h"

template <typename V, typename S, bool G, bool P>
UtilityClass<V, S, G, P>::UtilityClass() {}

template <typename V, typename S, bool G, bool P>
UtilityClass<V, S, G, P>::~UtilityClass() {}

template <typename V, typename S, bool G, bool P> template<typename T>
void UtilityClass<V, S, G, P>::AdjustComputationPointers(ValueType* basePtr, std::vector<T*>& vec, std::vector<bool>& isRootContainer, size_t* ValuesRootOffset) {
	SizeType numData = 0;

	std::vector<bool>::iterator isRoot = isRootContainer.begin();
	for (typename std::vector<T*>::iterator it = vec.begin(), it_e = vec.end(); it != it_e; ++it, ++isRoot) {
		if (!*isRoot) {
			(*it)->SetValueSize(basePtr + numData, (*it)->GetSizePtr(), (*it)->GetNumDim());
			numData += (*it)->GetNumEl();
		}
	}
	*ValuesRootOffset = numData;
	isRoot = isRootContainer.begin();
	for (typename std::vector<T*>::iterator it = vec.begin(), it_e = vec.end(); it != it_e; ++it, ++isRoot) {
		if (*isRoot) {
			(*it)->SetValueSize(basePtr + numData, (*it)->GetSizePtr(), (*it)->GetNumDim());
			numData += (*it)->GetNumEl();
		}
	}
}

template <typename V, typename S, bool G, bool P> template <typename T>
int UtilityClass<V, S, G, P>::UpdateCPUDataOffset(ValueType* basePtr, std::vector<T*>& vec, std::vector<bool>& isRootContainer, bool ignoreNonRoot) {
	SizeType numData = 0;

	std::vector<bool>::iterator isRoot = isRootContainer.begin();
	if (!ignoreNonRoot) {
		for (typename std::vector<T*>::iterator it = vec.begin(), it_e = vec.end(); it != it_e; ++it, ++isRoot) {
			if (!*isRoot) {
				(*it)->SetCPUDataOffset(basePtr + numData);
				numData += (*it)->GetNumEl();
			}
		}
		isRoot = isRootContainer.begin();
	}
	for (typename std::vector<T*>::iterator it = vec.begin(), it_e = vec.end(); it != it_e; ++it, ++isRoot) {
		if (*isRoot) {
			(*it)->SetCPUDataOffset(basePtr + numData);
			numData += (*it)->GetNumEl();
		}
	}
	return 0;
}

template <typename V, typename S, bool G, bool P> template <typename T>
void UtilityClass<V, S, G, P>::AdjustDerivativePointers(ValueType* baseDataPtr, std::vector<T*>& vec, std::vector<bool>& isRootContainer, ValueType* baseCPUDerivativeRootPtr) {
	SizeType numData = 0;
	std::vector<bool>::iterator isRoot = isRootContainer.begin();
	for (typename std::vector<T*>::iterator it = vec.begin(), it_e = vec.end(); it != it_e; ++it, ++isRoot) {
		if (!*isRoot) {
			ValueType** output = (*it)->GetDiffGradientAndEmpMean();
			assert(output != NULL && *output == NULL);
			*output = baseDataPtr + numData;
			numData += (*it)->GetNumEl();
		}
	}
	isRoot = isRootContainer.begin();
	for (typename std::vector<T*>::iterator it = vec.begin(), it_e = vec.end(); it != it_e; ++it, ++isRoot) {
		if (*isRoot) {
			ValueType** output = (*it)->GetDiffGradientAndEmpMean();
			assert(output != NULL && *output == NULL);
			*output = baseDataPtr + numData;
			SizeType buf = (*it)->GetNumEl();
			numData += buf;

			(*it)->SetCPUDerivativeRootPtr(baseCPUDerivativeRootPtr);
			baseCPUDerivativeRootPtr += buf;
		}
	}
}

template class UtilityClass<double, int, false, false>;
template class UtilityClass<double, int, true, false>;
template class UtilityClass<float, int, false, false>;
template class UtilityClass<float, int, true, false>;

#define INSTANTIATION_ADJUSTPOINTERS(x) \
	template void UtilityClass<double, int, false, false>::AdjustComputationPointers<x<Node<double, int, false> > >(ValueType*, std::vector<x<Node<double, int, false> >*>&, std::vector<bool>&, size_t*); \
	template void UtilityClass<double, int, true, false>::AdjustComputationPointers<x<Node<double, int, true> > >(ValueType*, std::vector<x<Node<double, int, true> >*>&, std::vector<bool>&, size_t*); \
	template void UtilityClass<float, int, false, false>::AdjustComputationPointers<x<Node<float, int, false> > >(ValueType*, std::vector<x<Node<float, int, false> >*>&, std::vector<bool>&, size_t*); \
	template void UtilityClass<float, int, true, false>::AdjustComputationPointers<x<Node<float, int, true> > >(ValueType*, std::vector<x<Node<float, int, true> >*>&, std::vector<bool>&, size_t*);

#define INSTANTIATION_UPDATECPUOFFSET(x) \
	template int UtilityClass<double, int, false, false>::UpdateCPUDataOffset<x<Node<double, int, false> > >(ValueType*, std::vector<x<Node<double, int, false> >*>&, std::vector<bool>&, bool); \
	template int UtilityClass<double, int, true, false>::UpdateCPUDataOffset<x<Node<double, int, true> > >(ValueType*, std::vector<x<Node<double, int, true> >*>&, std::vector<bool>&, bool); \
	template int UtilityClass<float, int, false, false>::UpdateCPUDataOffset<x<Node<float, int, false> > >(ValueType*, std::vector<x<Node<float, int, false> >*>&, std::vector<bool>&, bool); \
	template int UtilityClass<float, int, true, false>::UpdateCPUDataOffset<x<Node<float, int, true> > >(ValueType*, std::vector<x<Node<float, int, true> >*>&, std::vector<bool>&, bool);

#define INSTANTIATION_ADJUSTDERIVATIVEPOINTER(x) \
	template void UtilityClass<double, int, false, false>::AdjustDerivativePointers<x<Node<double, int, false> > >(ValueType*, std::vector<x<Node<double, int, false> >*>&, std::vector<bool>&, ValueType*); \
	template void UtilityClass<double, int, true, false>::AdjustDerivativePointers<x<Node<double, int, true> > >(ValueType*, std::vector<x<Node<double, int, true> >*>&, std::vector<bool>&, ValueType*); \
	template void UtilityClass<float, int, false, false>::AdjustDerivativePointers<x<Node<float, int, false> > >(ValueType*, std::vector<x<Node<float, int, false> >*>&, std::vector<bool>&, ValueType*); \
	template void UtilityClass<float, int, true, false>::AdjustDerivativePointers<x<Node<float, int, true> > >(ValueType*, std::vector<x<Node<float, int, true> >*>&, std::vector<bool>&, ValueType*);


INSTANTIATION_ADJUSTPOINTERS(Data)
INSTANTIATION_ADJUSTPOINTERS(Parameters)
INSTANTIATION_ADJUSTPOINTERS(ComputeFunction)

INSTANTIATION_UPDATECPUOFFSET(Data)
INSTANTIATION_UPDATECPUOFFSET(Parameters)
INSTANTIATION_UPDATECPUOFFSET(ComputeFunction)

INSTANTIATION_ADJUSTDERIVATIVEPOINTER(Parameters)
INSTANTIATION_ADJUSTDERIVATIVEPOINTER(ComputeFunction)