#ifndef DPHEADER
#define DPHEADER
#include <algorithm>
#include <vector>
using std::vector;

template<bool positive=false>
double DP(const vector<double>&, const vector<double>&, vector<size_t>&, const double);
#endif // DP


