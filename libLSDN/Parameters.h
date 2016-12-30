//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include "Node.h"

template <class N = Node<double, int, false> >
class Parameters : public N {
public:
	typedef N NodeType;
	typedef typename N::ValueType ValueType;
	typedef typename N::SizeType SizeType;
	typedef typename N::TreePostIter TreePostIter;

	struct NodeParameters {
		ValueType stepLength;
		ValueType momentum;
		ValueType stepDivider;
		ValueType l2_reg;
		ValueType norm_limit;
		void (*UpdateCallback)(Parameters*);
		NodeParameters(ValueType stepLength, ValueType moment, ValueType stepDivider, ValueType l2_reg, ValueType norm_limit, void (*UpdateCallback)(Parameters*)) : stepLength(stepLength), momentum(moment), stepDivider(stepDivider), l2_reg(l2_reg), norm_limit(norm_limit), UpdateCallback(UpdateCallback) {};
	};
protected:
	ValueType* DiffGradNEmpMean;
	ValueType* DiffHist;
	ValueType stepLength;
	ValueType moment;
	ValueType stepDivider;
	ValueType l2_reg;
	ValueType norm_limit;
	void(*UpdateCallback)(Parameters*);
public:
	Parameters(const NodeParameters&);
	virtual ~Parameters();

	virtual NODETYPE IdentifyMe();
	virtual void Clear();
	virtual void DeletedByContainer();
	virtual ValueType** GetDiffGradientAndEmpMean();
	virtual ValueType** GetDiffHist();

	virtual void Evaluate(TreePostIter&, STATE);
	virtual void Gradient(TreePostIter&);
	virtual void UpdateParameters();

	virtual void ReduceStepSize();
	virtual ValueType GetRegularization();
};

#endif