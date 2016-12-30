#ifndef CNNHEADER
#define CNNHEADER

#include <numeric>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
//#include <Eigen/Core>

//#define cimg_display 0
//#define cimg_use_jpeg
//#define cimg_use_png
//#define cimg_use_openmp
//#include "CImg.h"

#include "cuda_runtime.h"

#include "../libLSDN/LSDN.h"
#include "../libLSDN/Parameters.h"
#include "../libLSDN/Data.h"
#include "../libLSDN/ComputationTree.h"
#include "../libLSDN/ParameterContainer.h"

#include "../libLSDN/Function_Affine.h"
#include "../libLSDN/Function_Relu.h"
#include "../libLSDN/Function_Dropout.h"
#include "../libLSDN/Function_Sigmoid.h"
#include "../libLSDN/Function_Softmax.h"
#include "../libLSDN/Function_Pool.h"
#include "../libLSDN/Function_Conv.h"
#include "../libLSDN/Function_ConvSub.h"
#include "../libLSDN/Function_Lrn.h"
#include "../libLSDN/Function_Interpolate.h"
#include "../libLSDN/Function_InterpolateToSize.h"

#include "../libLSDN/LSDN_mathfunctions.h"

//#include "../ReadDirectory.h"
#include "../CPrecisionTimer.h"
#include "Conf.h"
#include <string>
//typedef Matrix<short, Dynamic, 1> VectorXs;
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
#ifdef _MSC_VER
#else
#endif

#ifdef _MSC_VER
#else
#endif

typedef Node<ValueType, SizeType, GPUTYPE> NodeType;
typedef Parameters<NodeType> ParaType;
typedef Data<NodeType> DataType;
typedef ComputeFunction<NodeType> FuncType;
typedef ComputationTree<NodeType> CompTree;

typedef vector<vector<vector<ValueType>>> vector3d;
typedef vector<vector<ValueType>> vector2d;

void SetWeights(const std::string& weightFile, ParameterContainer<ValueType, SizeType, GPUTYPE, false>& paramContainer);
void initCNN(const double alpha, const double beta, const int GPUid = 0);
void deleteCNN(CompTree *DeepNet16);

template<bool positive=true>
pair<double,double> APDLMTrain(string mode, const double epsilon);

template<bool positive=true>
void train(string mode, const double epsilon);

pair<double,double> APSVMTrain(string mode);
double LogisticTrain(string mode);
double HingeTrain(string mode);
void test(string testset);
#endif // CNN

