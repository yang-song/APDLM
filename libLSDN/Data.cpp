//Author: Alexander G. Schwing (http://alexander-schwing.de)
#include "Data.h"

template <class N>
Data<N>::Data(const NodeParameters&) {}

template <class N>
Data<N>::~Data() {}

template <class N>
void Data<N>::Clear() {
	NodeType::Clear();
}

template <class N>
void Data<N>::Evaluate(TreePostIter&, STATE) {}

template <class N>
void Data<N>::Gradient(TreePostIter&) {}

template class Data<Node<double, int, false> >;
template class Data<Node<double, int, true> >;
template class Data<Node<float, int, false> >;
template class Data<Node<float, int, true> >;