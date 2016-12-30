//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __FUNCTION_RELU_H__
#define __FUNCTION_RELU_H__
#include <algorithm>
#include <functional>

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class ReluFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {};
private:
	//virtual void Evaluate(i2t<true>, TreePostIter cur, STATE);
	//virtual void Evaluate(i2t<false>, TreePostIter cur, STATE);
	virtual void Gradient(i2t<true>, TreePostIter& cur);
	virtual void Gradient(i2t<false>, TreePostIter& cur);
public:
	ReluFunction(const NodeParameters&);
	virtual ~ReluFunction();

	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);

	static void PerformReluForward(i2t<false>, ValueType* memIn, SizeType num, ValueType* memOut);
	static void PerformReluForward(i2t<true>, ValueType* memIn, SizeType num, ValueType* memOut);
	static void PerformReluBackwardNoAdd(i2t<false>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut);
	static void PerformReluBackwardNoAdd(i2t<true>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut);
	static void PerformReluBackwardAdd(i2t<false>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut);
	static void PerformReluBackwardAdd(i2t<true>, ValueType* memIn, ValueType* memCheckForLTZero, SizeType num, ValueType* memOut);
};

#endif
