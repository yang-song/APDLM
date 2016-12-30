//Author: Alexander G. Schwing (http://alexander-schwing.de)
//Author: Liang-Chieh (Jay) Chen (http://www.cs.ucla.edu/~lcchen/)
#ifndef __FUNCTION_CONV_H__
#define __FUNCTION_CONV_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class ConvFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {
		SizeType p;
		SizeType s;
		SizeType g;
		bool hasBias;
		bool hasRELU;
		NodeParameters(SizeType p, SizeType s, SizeType g, bool b, bool r) : p(p), s(s), g(g), hasBias(b), hasRELU(r) {};
	};

private:
	ValueType* patchMatrix_;
	ValueType* patchMatrixDiff_;
	SizeType* sz_pm_;   //always 2 dimensions
	SizeType padSize_;
	SizeType stride_;
	SizeType numGroup_;
	bool hasBias_;
	bool hasRELU_;

	void Im2Patches(i2t<true>,  ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol);
	void Im2Patches(i2t<false>, ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol);
	void AccumPatchDiff2Im(i2t<true>,  ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol);
	void AccumPatchDiff2Im(i2t<false>, ValueType* img, SizeType numRow, SizeType numCol, SizeType numChan, SizeType kNumRow, SizeType kNumCol);
	void AdditionModuloOperand(i2t<false>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo);
	void AdditionModuloOperand(i2t<true>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo);
	void BiasDerivativeSingleDim(i2t<false>, ValueType* res, ValueType* input, SizeType patchSize, SizeType numSamples, SizeType numChannels, bool performAddition);
	void BiasDerivativeSingleDim(i2t<true>, ValueType* res, ValueType* input, SizeType patchSize, SizeType numSamples, SizeType numChannels, bool performAddition);
	void BiasDerivativeMultiDim(i2t<false>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition);
	void BiasDerivativeMultiDim(i2t<true>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition);
	void PerformInPlaceReluForward(i2t<false>, ValueType* mem, ValueType num);
	void PerformInPlaceReluForward(i2t<true>, ValueType* mem, ValueType num);
public:
	ConvFunction(const NodeParameters&);
	virtual ~ConvFunction();

	virtual void Clear();
	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};

#endif
