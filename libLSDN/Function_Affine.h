//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __FUNCTION_AFFINE_H__
#define __FUNCTION_AFFINE_H__

#include "ComputeFunction.h"

template <class N = Node<double, int, false> >
class AffineFunction : public ComputeFunction<N> {
public:
	typedef typename ComputeFunction<N>::ValueType ValueType;
	typedef typename ComputeFunction<N>::SizeType SizeType;
	typedef typename ComputeFunction<N>::TreePostIter TreePostIter;
	typedef typename ComputeFunction<N>::NodeType NodeType;

	struct NodeParameters {
		bool hasBias;
		bool hasRelu;
		NodeParameters(bool b, bool r) : hasBias(b), hasRelu(r) {};
	};
private:
	bool hasBias_;
	bool hasRelu_;

	void AdditionModuloOperand(i2t<false>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo);
	void AdditionModuloOperand(i2t<true>, ValueType* res, SizeType numEl, ValueType* addend, SizeType op_division, SizeType op_modulo);
	void BiasDerivativeSingleDim(i2t<false>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition);
	void BiasDerivativeSingleDim(i2t<true>, ValueType* res, ValueType* input, SizeType sampleSize, SizeType numSamples, bool performAddition);
public:
	AffineFunction(const NodeParameters&);
	virtual ~AffineFunction();

	virtual void Evaluate(TreePostIter& cur, STATE);
	virtual void Gradient(TreePostIter& cur);

	virtual void AdjustDimension(TreePostIter& cur);

	virtual ValueType GetValue(size_t ix, int AccessOffset, int Stride);
};

#endif