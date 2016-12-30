//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __FUNCTION_INTERPOLATETOSIZE_H__
#define __FUNCTION_INTERPOLATETOSIZE_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class InterpolateToSizeFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {
		SizeType out_h;
		SizeType out_w;
		NodeParameters(SizeType kh, SizeType kw) : out_h(kh), out_w(kw) {};
	};
private:
	SizeType out_h_;
	SizeType out_w_;

	void BilinearInterpolateForward(i2t<true>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out);
	void BilinearInterpolateForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out);
	void BilinearInterpolateBackward(i2t<true>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out);
	void BilinearInterpolateBackward(i2t<false>, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, SizeType numel_bot, ValueType* top_diff, SizeType h_out, SizeType w_out);

public:
	InterpolateToSizeFunction(const NodeParameters&);
	virtual ~InterpolateToSizeFunction();

	virtual void Clear();
	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};


#endif /* FUNCTION_POOLING_H_ */
