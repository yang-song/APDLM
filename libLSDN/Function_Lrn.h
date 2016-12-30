//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifndef __FUNCTION_LRN_H__
#define __FUNCTION_LRN_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class LrnFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {
		SizeType l;
		ValueType a;
		ValueType b;
		NodeParameters(SizeType l, ValueType a, ValueType b) : l(l), a(a), b(b) {};
	};
private:
	SizeType lrn_size_;  //size for local response normalization
	ValueType alpha_;
	ValueType beta_;

	ValueType* scale_data_;  //denominator of the normalization


	void LrnAcrossChannelForward(i2t<true>,  ValueType* in);
	void LrnAcrossChannelForward(i2t<false>, ValueType* in);

	void LrnAcrossChannelBackward(i2t<true>,  ValueType* val1, ValueType* output);
	void LrnAcrossChannelBackward(i2t<false>, ValueType* val1, ValueType* output);

public:
	LrnFunction(const NodeParameters&);
	virtual ~LrnFunction();

	virtual void Clear();
	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};


#endif
