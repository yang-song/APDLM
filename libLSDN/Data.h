//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __DATA_H__
#define __DATA_H__

#include "Node.h"

template <class N = Node<double, int, false> >
class Data : public N {
public:
	typedef N NodeType;
	typedef typename N::ValueType ValueType;
	typedef typename N::SizeType SizeType;
	typedef typename N::TreePostIter TreePostIter;

	struct NodeParameters {};
public:
	Data(const NodeParameters&);
	virtual ~Data();

	virtual void Clear();

	virtual void Evaluate(TreePostIter&, STATE);
	virtual void Gradient(TreePostIter&);
};

#endif