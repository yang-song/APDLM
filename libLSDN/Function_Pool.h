//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifndef __FUNCTION_POOLING_H__
#define __FUNCTION_POOLING_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class PoolFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	enum POOL_METHOD : unsigned char { MAX_POOLING, AVG_POOLING };

	struct NodeParameters {
		SizeType kh;
		SizeType kw;
		SizeType p;
		SizeType s;
		SizeType SubsampleH;
		SizeType SubsampleW;
		POOL_METHOD t;
		NodeParameters(SizeType kh, SizeType kw, SizeType p, SizeType s, SizeType subsampleH, SizeType subsampleW, POOL_METHOD t) : kh(kh), kw(kw), p(p), s(s), SubsampleH(subsampleH), SubsampleW(subsampleW), t(t) {};
	};
private:
	SizeType kernel_h_;
	SizeType kernel_w_;
	SizeType padSize_;
	SizeType stride_;
	SizeType SubsampleH_;
	SizeType SubsampleW_;
	POOL_METHOD pool_method_;  //0:max, 1:avg

	void MaxPoolForward(i2t<true>,  ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out);
	void MaxPoolForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out);
	void MaxPoolBackward(i2t<true>, ValueType* bot_data, ValueType* bot_diff, SizeType h_bot, SizeType w_bot, ValueType* top_data, ValueType* top_diff, SizeType h_out, SizeType w_out);
	void MaxPoolBackward(i2t<false>,ValueType* bot_data_pt, ValueType* bot_diff_pt, SizeType h_bot, SizeType w_bot, ValueType* top_data_pt, ValueType* top_diff_pt, SizeType h_out, SizeType w_out);

	void AvgPoolForward(i2t<true>,  ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out);
	void AvgPoolForward(i2t<false>, ValueType* bottom_pt, SizeType h_bot, SizeType w_bot, ValueType* top_pt, SizeType h_out, SizeType w_out);
	void AvgPoolBackward(i2t<true>, ValueType* bot_diff_pt, SizeType h_bot, SizeType w_bot, ValueType* top_diff_pt, SizeType h_out, SizeType w_out);
	void AvgPoolBackward(i2t<false>,ValueType* bot_diff_pt, SizeType h_bot, SizeType w_bot, ValueType* top_diff_pt, SizeType h_out, SizeType w_out);

public:
	PoolFunction(const NodeParameters&);
	virtual ~PoolFunction();

	virtual void Clear();
	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};


#endif /* FUNCTION_POOLING_H_ */
