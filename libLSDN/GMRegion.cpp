#ifndef LSDN_USE_GRAPH
#include "GMRegion.h"

template <class T>
GMRegion<T>::GMRegion() : pot(NULL) {}

template <class T>
GMRegion<T>::~GMRegion() {}

template class GMRegion<ComputationTree<Node<double, int, false> > >;
template class GMRegion<ComputationTree<Node<double, int, true> > >;
template class GMRegion<ComputationTree<Node<float, int, false> > >;
template class GMRegion<ComputationTree<Node<float, int, true> > >;

template class GMRegion<AccessPotential<Node<double, int, false> > >;
template class GMRegion<AccessPotential<Node<double, int, true> > >;
template class GMRegion<AccessPotential<Node<float, int, false> > >;
template class GMRegion<AccessPotential<Node<float, int, true> > >;
#endif