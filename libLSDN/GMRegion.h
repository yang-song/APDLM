#ifndef __GMREGION_H__
#define __GMREGION_H__
#ifndef LSDN_USE_GRAPH
#include "ComputationTree.h"


template <class T=Node<double, int, false> >
class AccessPotential {
public:
	typedef T NodeType;
	typedef typename T::ValueType ValueType;
	typedef typename T::SizeType SizeType;
private:
	ValueType* ptr;
	NodeType* n;
public:
	AccessPotential(ValueType* data, NodeType* node) : ptr(data), n(node) {};
	virtual ~AccessPotential() {};
	virtual ValueType operator()(size_t ix, int AccessOffset, int Stride) {
		return ptr[ix*Stride + AccessOffset];
	}
	virtual NodeType* GetRoot() {
		return n;
	}
	virtual ValueType* GetPtr() {
		return ptr;
	}
	virtual void SetPtr(ValueType* p1, NodeType* n1) {
		ptr = p1;
		n = n1;
	}
};

template <class T = ComputationTree<Node<double, int, false> > >
class GMRegion {
public:
	typedef T TreeType;
	typedef typename TreeType::ValueType ValueType;
	typedef typename TreeType::SizeType SizeType;
public:
	ValueType c_r;
	TreeType* pot;
	ValueType* bel;
	int* num_states;
	int* cum_num_states;
	int* var_ix;
	int num_variables;
	int Color;
	int flag;
	SizeType AccessOffset;
	SizeType Stride;
	int CompTreeID;
public:
	GMRegion();
	~GMRegion();
};

#endif
#endif