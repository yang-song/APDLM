//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __FUNCTION_DROPOUT_H__
#define __FUNCTION_DROPOUT_H__
#include <functional>
#include <algorithm>

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class DropoutFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {
		ValueType dropout_rate;
		NodeParameters(ValueType rate) : dropout_rate(rate) {};
	};

private:
	ValueType dropout_rate_;
	ValueType scale_;
	unsigned int* mask_;

	void rngUniform(i2t<false>, const SizeType numEl, unsigned int* mask);
	void rngUniform(i2t<true>, const SizeType numEl, unsigned int* mask);
	void MaskOperation(i2t<false>, ValueType* input, ValueType* output, unsigned int threshold, SizeType Length, bool addToOutput);
	void MaskOperation(i2t<true>, ValueType* input, ValueType* output, unsigned int threshold, SizeType Length, bool addToOutput);
public:
	DropoutFunction(const NodeParameters&);
	virtual ~DropoutFunction();

	virtual void Clear();
	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual unsigned int* AllocMaskMem(i2t<true>, SizeType numEl);
	virtual unsigned int* AllocMaskMem(i2t<false>, SizeType numEl);
	virtual void DeAllocMaskMem(i2t<true>, unsigned int* ptr);
	virtual void DeAllocMaskMem(i2t<false>, unsigned int* ptr);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};

#endif