//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifndef __FUNCTION_SOFTMAX_H__
#define __FUNCTION_SOFTMAX_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class SoftmaxFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {};
private:
	ValueType* scale_val;
	ValueType* all_one;
	SizeType numEl_AllButOne;
	
	void GetMax(i2t<false>);
	void GetMax(i2t<true>);
	void DiagonalInnerProduct(i2t<false>);
	void DiagonalInnerProduct(i2t<true>);
	void ElementwiseMultiply(i2t<false>, SizeType dim, ValueType* data_in1, ValueType* data_in2, ValueType* data_out, bool addToOutput);
	void ElementwiseMultiply(i2t<true>, SizeType dim, ValueType* data_in1, ValueType* data_in2, ValueType* data_out, bool addToOutput);
public:
	SoftmaxFunction(const NodeParameters&);
	virtual ~SoftmaxFunction();

	virtual void Clear();
	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};

#endif