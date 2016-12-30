//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __FUNCTION_SIGMOID_H__
#define __FUNCTION_SIGMOID_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class SigmoidFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {};
private:
	void SigmoidForward(i2t<false>, ValueType* input);
	void SigmoidForward(i2t<true>, ValueType* input);
	void SigmoidBackward(i2t<false>, ValueType* output, bool addToOutput);
	void SigmoidBackward(i2t<true>, ValueType* output, bool addToOutput);
public:
	SigmoidFunction(const NodeParameters&);
	virtual ~SigmoidFunction();

	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};

#endif