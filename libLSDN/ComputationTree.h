//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __COMPUTATIONTREE_H__
#define __COMPUTATIONTREE_H__
#ifndef LSDN_USE_GRAPH

#include <set>

#include "Node.h"

template <class N = Node<double, int, false> >
class ComputationTree {
public:
	typedef N NodeType;
	typedef typename NodeType::ValueType ValueType;
	typedef typename NodeType::SizeType SizeType;
	typedef typename NodeType::TreePostIter TreePostIter;
	typedef typename NodeType::TreeSiblIter TreeSiblIter;
private:
	tree<NodeType*> tree_;
public:
	ComputationTree();
	virtual ~ComputationTree();

	virtual void Clear(std::set<NodeType*>* ToBeDeleted, bool DontTouchParams);
	virtual ValueType operator()(size_t ix, int AccessOffset, int Stride);
	virtual void ForwardPass(STATE state);
	virtual void ForwardPassAdjustDimension();
	virtual void BackwardPass();
	virtual TreeSiblIter insert(NodeType* nd);
	virtual TreeSiblIter append_child(NodeType* nd, TreeSiblIter& below);
	virtual NodeType* GetRoot();
};

#endif
#endif