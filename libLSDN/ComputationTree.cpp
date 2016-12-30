//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef LSDN_USE_GRAPH
#include "ComputationTree.h"

template <class N>
ComputationTree<N>::ComputationTree() {}

template <class N>
ComputationTree<N>::~ComputationTree() {}

template <class N>
void ComputationTree<N>::Clear(std::set<NodeType*>* ToBeDeleted, bool DontTouchParams) {
	for (typename tree<NodeType*>::post_order_iterator it = tree_.begin_post(), it_e = tree_.end_post(); it != it_e; ++it) {
		if (!(DontTouchParams == true && (*it)->IdentifyMe() == NODE_PARAM)) {
			(*it)->Clear();
			if (ToBeDeleted != NULL) {
				ToBeDeleted->insert(*it);
			}
		}
	}
}

template <class N>
typename ComputationTree<N>::ValueType ComputationTree<N>::operator()(size_t ix, int AccessOffset, int Stride) {
	return (*tree_.begin())->GetValue(ix, AccessOffset, Stride);
}

template <class N>
void ComputationTree<N>::ForwardPass(STATE state) {
	for (typename tree<NodeType*>::post_order_iterator it = tree_.begin_post(), it_e = tree_.end_post(); it != it_e; ++it) {
		//(*it)->PrintName();
		(*it)->Evaluate(it, state);
	}
}

template <class N>
void ComputationTree<N>::ForwardPassAdjustDimension() {
	for (typename tree<NodeType*>::post_order_iterator it = tree_.begin_post(), it_e = tree_.end_post(); it != it_e; ++it) {
		(*it)->AdjustDimension(it);
	}
}

template <class N>
void ComputationTree<N>::BackwardPass() {
	for (typename tree<NodeType*>::pre_order_iterator it = tree_.begin(), it_e = tree_.end(); it != it_e; ++it) {
		typename tree<NodeType*>::post_order_iterator tmp(it);
		(*it)->Gradient(tmp);
	}
}

template <class N>
typename ComputationTree<N>::TreeSiblIter ComputationTree<N>::insert(NodeType* nd) {
	return tree_.insert(tree_.begin(), nd);
}

template <class N>
typename ComputationTree<N>::TreeSiblIter ComputationTree<N>::append_child(NodeType* nd, typename ComputationTree<N>::TreeSiblIter& below) {
	return tree_.append_child(below, nd);
}

template <class N>
typename ComputationTree<N>::NodeType* ComputationTree<N>::GetRoot() {
	return *tree_.begin();
}

template class ComputationTree<Node<double, int, false> >;
template class ComputationTree<Node<double, int, true> >;
template class ComputationTree<Node<float, int, false> >;
template class ComputationTree<Node<float, int, true> >;
#endif