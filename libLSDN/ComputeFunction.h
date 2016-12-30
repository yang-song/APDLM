//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __COMPUTEFUNCTION_H__
#define __COMPUTEFUNCTION_H__

#include "Node.h"

template <class N = Node<double, int, false> >
class ComputeFunction : public N {
public:
	typedef N NodeType;
	typedef typename N::ValueType ValueType;
	typedef typename N::SizeType SizeType;
	typedef typename N::TreePostIter TreePostIter;
protected:
	ValueType* DiffGradNEmpMean;
public:
	ComputeFunction();
	virtual ~ComputeFunction();
	virtual void Clear();
	virtual void DeletedByContainer();

	virtual ValueType** GetDiffGradientAndEmpMean();

	virtual void Evaluate(TreePostIter&, STATE);
	virtual void Gradient(TreePostIter&);
};

#endif