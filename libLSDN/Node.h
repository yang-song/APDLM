//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __NODE_H__
#define __NODE_H__

#include <string>

#include "../LSDN_Common.h"

#ifdef LSDN_USE_GRAPH

#else
#include "tree.hh"
#define LSDN_NUMBER_OF_CHILDREN number_of_children
#define LSDN_NODE_ACCESSOR(x) (*(x))
#endif

template <class V = double, class S = int, bool G = false>
class Node {
public:
	typedef V ValueType;
	typedef S SizeType;
#ifdef LSDN_USE_GRAPH

#else
	typedef typename tree<Node<V, S, G>*>::post_order_iterator	TreePostIter;
	typedef typename tree<Node<V, S, G>*>::sibling_iterator		TreeSiblIter;
#endif
	std::string name;
protected:
	ValueType* value;
	ValueType* CPUDataOffset;
	ValueType* CPUDerivativeRootPtr;
	SizeType* sz;
	SizeType numDim;
	SizeType numEl;
	bool isMyValue;
public:
	typedef i2t<G> GPUType;
public:
	Node();
	virtual ~Node();

	virtual void SetName(const char* n);
	virtual void PrintName();
	virtual void Clear();
	virtual void DeletedByContainer();
	virtual NODETYPE IdentifyMe();
	virtual ValueType* GetValuePtr();
	virtual SizeType* GetSizePtr();
	virtual SizeType GetNumDim();
	virtual SizeType GetNumEl();
	virtual void SetValueSize(ValueType* v, SizeType* s, SizeType dim, bool transferValueOwnership = true);
	virtual SizeType ComputeNumEl();
	virtual ValueType** GetDiffGradientAndEmpMean();
	virtual ValueType** GetDiffHist();
	virtual void AdjustDimension(TreePostIter&);

	virtual void Evaluate(TreePostIter&, STATE) = 0;
	virtual void Gradient(TreePostIter&) = 0;

	virtual ValueType*  AllocValueMem(i2t<true>, SizeType numEl);
	virtual ValueType*  AllocValueMem(i2t<false>, SizeType numEl);
	virtual void DeAllocValueMem(i2t<true>, ValueType* ptr);
	virtual void DeAllocValueMem(i2t<false>, ValueType* ptr);

	virtual ValueType GetValue(size_t, int, int);

	virtual void SetCPUDataOffset(ValueType* val);
	virtual ValueType* GetCPUDataOffset();
	virtual void SetCPUDerivativeRootPtr(ValueType* val);
	virtual ValueType* GetCPUDerivativeRootPtr();
};

#endif
