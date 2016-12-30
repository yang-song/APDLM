/* Code originally comes from DeepDenseCRF by Alex Schwing (aschwing@illinois.edu)
 * Adapted by Yang Song (yangsong@cs.stanford.edu)
 * This file contains the codes for
 * 1. Neural network architecture and initialization
 * 2. Direct loss minimization training function APDLMTrain<bool>()
 * 3. AP-SVM training function APSVMTrain()
 * 4. Point wise hinge loss training function HingeTrain()
 * 5. Point wise cross entropy training function LogisticTrain()
 *
 * The entrance to all training functions is train<bool>()
*/
#include "cnn.h"
#include <random>
#include "database.h"
#include "dp.h"
#include "Image.h"
#include <limits>
#include <cstring>
//using namespace Eigen;
#include <random>
#include <algorithm>
#include <utility>
#include <cassert>
#include <queue>
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
#define INF std::numeric_limits<int>::max()

CompTree::TreeSiblIter AppendAffineFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    AffineFunction<NodeType>* AffFunc = new AffineFunction<NodeType>(AffineFunction<NodeType>::NodeParameters(bias, relu));
    AffFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(AffFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}

CompTree::TreeSiblIter AppendConvFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu, SizeType padding) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    ConvFunction<NodeType>* ConvFunc = new ConvFunction<NodeType>(ConvFunction<NodeType>::NodeParameters(padding, 1, 1, bias, relu));
    ConvFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(ConvFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendConvSubFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu, SizeType padding, SizeType SubsampleH, SizeType SubsampleW) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    ConvSubFunction<NodeType>* ConvSubFunc = new ConvSubFunction<NodeType>(ConvSubFunction<NodeType>::NodeParameters(padding, 1, 1, SubsampleH, SubsampleW, bias, relu));
    ConvSubFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(ConvSubFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendDropoutFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, ValueType rate, NodeType* param1, NodeType* param2) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    DropoutFunction<NodeType>* DropFunc = new DropoutFunction<NodeType>(DropoutFunction<NodeType>::NodeParameters(rate));
    DropFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(DropFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendPoolingFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, SizeType stride, SizeType padding, SizeType SubsampleH, SizeType SubsampleW) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    PoolFunction<NodeType>* PoolFunc = new PoolFunction<NodeType>(PoolFunction<NodeType>::NodeParameters(2, 2, padding, stride, SubsampleH, SubsampleW, PoolFunction<NodeType>::MAX_POOLING));
    PoolFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(PoolFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
// Another version
CompTree::TreeSiblIter AppendPoolingFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, SizeType kernelheight, SizeType kernelwidth, SizeType stride, SizeType padding, SizeType SubsampleH, SizeType SubsampleW) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    PoolFunction<NodeType>* PoolFunc = new PoolFunction<NodeType>(PoolFunction<NodeType>::NodeParameters(kernelheight, kernelwidth, padding, stride, SubsampleH, SubsampleW, PoolFunction<NodeType>::MAX_POOLING));
    PoolFunc->SetValueSize(NULL, NULL, 2);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(PoolFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendLRNFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, SizeType length, ValueType alpha, ValueType beta) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    LrnFunction<NodeType>* LRNFunc = new LrnFunction<NodeType>(LrnFunction<NodeType>::NodeParameters(length, alpha, beta));
    LRNFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(LRNFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}
CompTree::TreeSiblIter AppendConvFunction(CompTree* CNN, CompTree::TreeSiblIter& nodeIter, NodeType* param1, NodeType* param2, bool bias, bool relu, SizeType padding, SizeType stride, SizeType groups) {
    if (param1 != NULL) {
        CNN->append_child(param1, nodeIter);
    }
    ConvFunction<NodeType>* ConvFunc = new ConvFunction<NodeType>(ConvFunction<NodeType>::NodeParameters(padding, stride, groups, bias, relu));
    ConvFunc->SetValueSize(NULL, NULL, 4);
    CompTree::TreeSiblIter nodeIter2 = CNN->append_child(ConvFunc, nodeIter);
    if (param2 != NULL) {
        CNN->append_child(param2, nodeIter);
    }
    return nodeIter2;
}

void CreateAlexNet(const double alpha, const double beta, CompTree* CNN, ParameterContainer<ValueType, SizeType, GPUTYPE, false>& paramContainer, DataType* dataC) {
//    ValueType stepSize1 = ValueType(-1e-6); //P5: ValueType(-5e-7); //P4: ValueType(-1e-5); //P3: ValueType(-1e-6); //P2: ValueType(-5e-5); //P1: ValueType(-5e-6);
//    ValueType stepSize2 = ValueType(-1e-7); //P5: ValueType(-5e-8); //P4: ValueType(-1e-6); //P3: ValueType(-1e-7); //P2: ValueType(-5e-6); //P1: ValueType(-5e-7);
    ValueType stepSize1 = ValueType(-alpha * 10) / (Conf::neg_num + Conf::pos_num);
    ValueType stepSize2 = ValueType(-alpha) / (Conf::neg_num + Conf::pos_num);
    ValueType moment = ValueType(0.9);// ValueType(0.9);
//    ValueType l2_reg = ValueType(0.000005);// ValueType(0.0005);// ValueType(1); // ValueType(0.0005);
    ValueType l2_reg = ValueType(beta) * (Conf::neg_num + Conf::pos_num);
    ValueType red = ValueType(10);

//    paramContainer.AddParameter(new SizeType[2]{4096, 76}, 2, ParaType::NodeParameters(stepSize1, moment, red, l2_reg, 0, NULL), false, 14);
    paramContainer.AddParameter(new SizeType[2]{4096, 1}, 2, ParaType::NodeParameters(stepSize1, moment, red, l2_reg, 0, NULL), false, 14);
    paramContainer.AddParameter(new SizeType[2]{4096, 4096}, 2, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 12);
    paramContainer.AddParameter(new SizeType[2]{9216, 4096}, 2, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 10);
    paramContainer.AddParameter(new SizeType[4]{3, 3, 192, 256}, 4, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 8);
    paramContainer.AddParameter(new SizeType[4]{3, 3, 192, 384}, 4, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 6);
    paramContainer.AddParameter(new SizeType[4]{3, 3, 256, 384}, 4, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 4);
    paramContainer.AddParameter(new SizeType[4]{5, 5, 48, 256}, 4, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 2);
    paramContainer.AddParameter(new SizeType[4]{11, 11, 3, 96}, 4, ParaType::NodeParameters(stepSize2, moment, red, l2_reg, 0, NULL), false, 0);
    paramContainer.AddParameter(new SizeType[1]{96}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 1);
    paramContainer.AddParameter(new SizeType[1]{256}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 3);
    paramContainer.AddParameter(new SizeType[1]{384}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 5);
    paramContainer.AddParameter(new SizeType[1]{384}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 7);
    paramContainer.AddParameter(new SizeType[1]{256}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 9);
    paramContainer.AddParameter(new SizeType[1]{4096}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 11);
    paramContainer.AddParameter(new SizeType[1]{4096}, 1, ParaType::NodeParameters(2 * stepSize2, moment, red, 0, 0, NULL), false, 13);
//    paramContainer.AddParameter(new SizeType[1]{76}, 1, ParaType::NodeParameters(stepSize1, moment, red, l2_reg, 0, NULL), false, 15);

    paramContainer.AddParameter(new SizeType[1]{1}, 1, ParaType::NodeParameters(2 * stepSize1, moment, red, 0, 0, NULL), false, 15);

    paramContainer.CreateCPUMemoryForParameters();
    paramContainer.PrepareComputation(TRAIN);

    AffineFunction<NodeType>* AffFunc = new AffineFunction<NodeType>(AffineFunction<NodeType>::NodeParameters(true, false));
    AffFunc->SetValueSize(NULL, NULL, 2);

    CompTree::TreeSiblIter nodeIter = CNN->insert(AffFunc);
    nodeIter = AppendDropoutFunction(CNN, nodeIter, ValueType(0.5), paramContainer.GetPtrFromID(14), paramContainer.GetPtrFromID(15));
    nodeIter = AppendAffineFunction(CNN, nodeIter, NULL, NULL, true, true);
    nodeIter = AppendDropoutFunction(CNN, nodeIter, ValueType(0.5), paramContainer.GetPtrFromID(12), paramContainer.GetPtrFromID(13));
    nodeIter = AppendAffineFunction(CNN, nodeIter, NULL, NULL, true, true);
    nodeIter = AppendPoolingFunction(CNN, nodeIter, paramContainer.GetPtrFromID(10), paramContainer.GetPtrFromID(11), 3, 3, 2, 0, 1, 1);
    nodeIter = AppendConvFunction(CNN, nodeIter, NULL, NULL, true, true, 1, 1, 2);
    nodeIter = AppendConvFunction(CNN, nodeIter, paramContainer.GetPtrFromID(8), paramContainer.GetPtrFromID(9), true, true, 1, 1, 2);
    nodeIter = AppendConvFunction(CNN, nodeIter, paramContainer.GetPtrFromID(6), paramContainer.GetPtrFromID(7), true, true, 1, 1, 1);
    nodeIter = AppendLRNFunction(CNN, nodeIter, paramContainer.GetPtrFromID(4), paramContainer.GetPtrFromID(5), 5, ValueType(0.0001), ValueType(0.75));
    nodeIter = AppendPoolingFunction(CNN, nodeIter, NULL, NULL, 3, 3, 2, 0, 1, 1);
    nodeIter = AppendConvFunction(CNN, nodeIter, NULL, NULL, true, true, 2, 1, 2);
    nodeIter = AppendLRNFunction(CNN, nodeIter, paramContainer.GetPtrFromID(2), paramContainer.GetPtrFromID(3), 5, ValueType(0.0001), ValueType(0.75));
    nodeIter = AppendPoolingFunction(CNN, nodeIter, NULL, NULL, 3, 3, 2, 0, 1, 1);
    nodeIter = AppendConvFunction(CNN, nodeIter, NULL, NULL, true, true, 0, 4, 1);

    CNN->append_child(paramContainer.GetPtrFromID(0), nodeIter);
    CNN->append_child(dataC, nodeIter);
    CNN->append_child(paramContainer.GetPtrFromID(1), nodeIter);
}

void SetWeights(const std::string& weightFile, ParameterContainer<ValueType, SizeType, GPUTYPE, false>& paramContainer) {
    std::vector<ValueType> initWeights(paramContainer.GetWeightDimension(), ValueType(0.0));
    std::ifstream ifs(weightFile.c_str(), std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
    if (ifs.is_open()) {
        std::ifstream::pos_type FileSize = ifs.tellg();
        ifs.seekg(0, std::ios_base::beg);
        if (size_t(FileSize) == initWeights.size()*sizeof(ValueType)) {
            ifs.read((char*)&initWeights[0], initWeights.size()*sizeof(ValueType));
        } else {
            std::cout << "Dimensions of initial weight file (" << FileSize / sizeof(ValueType) << ") don't match parameter dimension (" << initWeights.size() << "). Using default." << std::endl;
        }
        ifs.close();
    } else {
        std::cout << "Could not open weight file '" << weightFile << "'. Using default." << std::endl;
    }
    if (initWeights.size() > 0) {
        paramContainer.SetWeights(i2t<GPUTYPE>(), &initWeights);
    }
}
void DerivativeTest() {
    ParameterContainer<ValueType, SizeType, GPUTYPE, false> DiffTestParams;
    //DiffTestParams.AddParameter(new SizeType[4]{7, 7, 5, 6}, 4, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 26);
    //DiffTestParams.AddParameter(new SizeType[1]{6}, 1, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 27);
    //DiffTestParams.AddParameter(new SizeType[4]{10, 10, 5, 1}, 4, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 0);
    //DiffTestParams.AddParameter(new SizeType[2]{20, 1}, 2, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 2);
    DiffTestParams.AddParameter(new SizeType[2]{20, 20}, 2, ParaType::NodeParameters(-1.0, 0.0, 2, 0, 10, NULL), false, 2);
    DiffTestParams.CreateCPUMemoryForParameters();
    DiffTestParams.PrepareComputation(TRAIN);

    std::vector<ValueType> DiffTestWeights(DiffTestParams.GetWeightDimension(), ValueType(1.0));
    srand(1);
    for (std::vector<ValueType>::iterator iter = DiffTestWeights.begin(), iter_e = DiffTestWeights.end(); iter != iter_e; ++iter) {
        *iter = ValueType(rand()) / ValueType(RAND_MAX) - ValueType(0.5);
        //*iter = std::max((iter - DiffTestWeights.begin()) % 10, ((iter - DiffTestWeights.begin()) / 10) % 10);
    }
    //DiffTestWeights[0] = ValueType(200);
    DiffTestParams.SetWeights(i2t<GPUTYPE>(), &DiffTestWeights);

    ConvFunction<NodeType>* ConvTestFunc = new ConvFunction<NodeType>(ConvFunction<NodeType>::NodeParameters(1, 1, 1, true, false));
    ConvTestFunc->SetValueSize(NULL, NULL, 0);
    ConvSubFunction<NodeType>* ConvSubTestFunc = new ConvSubFunction<NodeType>(ConvSubFunction<NodeType>::NodeParameters(1, 1, 1, 4, 4, true, false));
    ConvSubTestFunc->SetValueSize(NULL, NULL, 0);
    AffineFunction<NodeType>* AffFunc = new AffineFunction<NodeType>(AffineFunction<NodeType>::NodeParameters(false, false));
    AffFunc->SetValueSize(NULL, NULL, 0);
    PoolFunction<NodeType>* PoolFunc = new PoolFunction<NodeType>(PoolFunction<NodeType>::NodeParameters(2, 2, 1, 1, 1, 1, PoolFunction<NodeType>::MAX_POOLING));
    PoolFunc->SetValueSize(NULL, NULL, 0);
    InterpolateFunction<NodeType>* IntPolFunc = new InterpolateFunction<NodeType>(InterpolateFunction<NodeType>::NodeParameters(3, 3));
    IntPolFunc->SetValueSize(NULL, NULL, 0);
    InterpolateToSizeFunction<NodeType>* IntPolSizeFunc = new InterpolateToSizeFunction<NodeType>(InterpolateToSizeFunction<NodeType>::NodeParameters(21, 21));
    IntPolSizeFunc->SetValueSize(NULL, NULL, 0);

    CompTree* DiffTest = new CompTree;
    CompTree::TreeSiblIter nodeIterTest = DiffTest->insert(AffFunc);
    DiffTest->append_child(DiffTestParams.GetPtrFromID(2), nodeIterTest);
    DiffTest->append_child(DiffTestParams.GetPtrFromID(2), nodeIterTest);
    DiffTest->append_child(DiffTestParams.GetPtrFromID(2), nodeIterTest);
    //DiffTest->append_child(DiffTestParams.GetPtrFromID(0), nodeIterTest);
    //DiffTest->append_child(DiffTestParams.GetPtrFromID(27), nodeIterTest);

    DiffTest->ForwardPass(TEST);
    SizeType numOutputEl = DiffTest->GetRoot()->GetNumEl();

    ValueType* DiffTestOutput;
    if (GPUTYPE) {
        DiffTestOutput = new ValueType[numOutputEl];
        cudaMemcpy((char*)DiffTestOutput, DiffTest->GetRoot()->GetValuePtr(), numOutputEl*sizeof(ValueType), cudaMemcpyDeviceToHost);
    } else {
        DiffTestOutput = DiffTest->GetRoot()->GetValuePtr();
    }

    ValueType SumRes = 0;
    for (SizeType k = 0; k < numOutputEl; ++k) {
        SumRes += DiffTestOutput[k];
    }

    ValueType* deriv = new ValueType[numOutputEl];
    std::fill(deriv, deriv + numOutputEl, ValueType(1.0));
    ValueType** diff = DiffTest->GetRoot()->GetDiffGradientAndEmpMean();
    ValueType* diffGPU = NULL;
    if (GPUTYPE) {
        cudaMalloc((void**)&diffGPU, sizeof(ValueType)*numOutputEl);
        cudaMemcpy(diffGPU, deriv, numOutputEl*sizeof(ValueType), cudaMemcpyHostToDevice);
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    DiffTest->BackwardPass();

    std::vector<ValueType> ComputedDerivative;
    DiffTestParams.GetDerivative(i2t<GPUTYPE>(), ComputedDerivative);

    std::vector<ValueType> ApproximatedDerivative(ComputedDerivative.size(), ValueType(0.0));
    std::vector<ValueType> ApproximatedDerivativeOnesided(ComputedDerivative.size(), ValueType(0.0));
    ValueType norm = ValueType(0.0);
    ValueType maxAbsDiff = ValueType(0.0);
    ValueType maxAbsDiffOS = ValueType(0.0);
    size_t maxDiffDim = 0;
    ValueType h = ValueType(1e-5);
    assert(ApproximatedDerivative.size() == DiffTestWeights.size());
    for (size_t k = 0; k < ApproximatedDerivative.size(); ++k) {
        ValueType origWeight = DiffTestWeights[k];
        DiffTestWeights[k] += h;

        DiffTestParams.SetWeights(i2t<GPUTYPE>(), &DiffTestWeights);
        DiffTest->ForwardPass(TEST);
        if (GPUTYPE) {
            cudaMemcpy((char*)DiffTestOutput, DiffTest->GetRoot()->GetValuePtr(), numOutputEl*sizeof(ValueType), cudaMemcpyDeviceToHost);
        }
        ValueType f1 = 0;
        for (SizeType m = 0; m < numOutputEl; ++m) {
            f1 += DiffTestOutput[m];
        }

        DiffTestWeights[k] = origWeight - h;

        DiffTestParams.SetWeights(i2t<GPUTYPE>(), &DiffTestWeights);
        DiffTest->ForwardPass(TEST);
        if (GPUTYPE) {
            cudaMemcpy((char*)DiffTestOutput, DiffTest->GetRoot()->GetValuePtr(), numOutputEl*sizeof(ValueType), cudaMemcpyDeviceToHost);
        }
        ValueType f2 = 0;
        for (SizeType m = 0; m < numOutputEl; ++m) {
            f2 += DiffTestOutput[m];
        }

        DiffTestWeights[k] = origWeight;

        ApproximatedDerivative[k] = (f1 - f2) / (2 * h);
        ApproximatedDerivativeOnesided[k] = (f1 - SumRes) / h;

        ValueType diff = ComputedDerivative[k] - ApproximatedDerivative[k];
        norm += (diff)*(diff);
        if (fabs(diff) > maxAbsDiff) {
            maxAbsDiff = fabs(diff);
            maxDiffDim = k;
        }

        ValueType diffOS = ComputedDerivative[k] - ApproximatedDerivativeOnesided[k];
        if (fabs(diffOS) > maxAbsDiffOS) {
            maxAbsDiffOS = fabs(diffOS);
        }
    }

    std::cout << "Norm of deriv diff:           " << std::sqrt(norm) << std::endl;
    std::cout << "Max abs value deriv diff:     " << maxAbsDiff << std::endl;
    std::cout << "Max abs value deriv diff dim: " << maxDiffDim << std::endl;
}

void deleteCNN(CompTree *DeepNet16){
    std::set<NodeType*> nothing;
    DeepNet16->Clear(&nothing, true);
    for (std::set<NodeType*>::iterator it = nothing.begin(), it_e = nothing.end(); it != it_e; ++it) {
        delete *it;
    }
    delete DeepNet16;
}

void initCNN(const double alpha, const double beta, const int GPUid) {

    if (GPUTYPE) {
        int GPUboard = GPUid;
        if (cudaSetDevice(GPUboard) != cudaSuccess) {
            std::cout << "Cannot set GPU device " << GPUboard << std::endl;
            return;
        } else {
            std::cout << "Using GPU " << GPUboard << std::endl;
        }
    } else {
        std::cout << "NOT using GPU. Are you sure?" << std::endl;
    }

    // Guess the random seed here is really important?
    LSDN::Instance().SetSeed(GPUid + 2015);
    database::data = new ValueType[Conf::crop_size * Conf::crop_size * 3 * Conf::batch_size]();

    if (GPUTYPE) {
        cudaMalloc((void**)&database::dataGPU, sizeof(ValueType)*Conf::crop_size * Conf::crop_size * 3 * Conf::batch_size);
        cudaMemcpy((char*)database::dataGPU, (char*)database::data, sizeof(ValueType)*Conf::crop_size * Conf::crop_size * 3 * Conf::batch_size
                   , cudaMemcpyHostToDevice);
        database::cnnData->SetValueSize(database::dataGPU, new SizeType[4]{Conf::crop_size,Conf::crop_size,3,Conf::batch_size}, 4);
    } else {
        database::cnnData->SetValueSize(database::data, new SizeType[4]{Conf::crop_size,Conf::crop_size,3,Conf::batch_size}, 4);
    }

    CreateAlexNet(alpha, beta, database::cnn, database::cnnParams, database::cnnData);
    database::loadname = Conf::dir_weights + Conf::load_weight;
    //database::loadname = Conf::dir_weights + "weights100.bak";
    database::loadWeights();

    database::cnn->ForwardPass(TRAIN);
    if(GPUTYPE){
        database::netOutput = new ValueType[database::cnn->GetRoot()->GetNumEl()];
    }else{
        database::netOutput = database::cnn->GetRoot()->GetValuePtr();
    }
    if (GPUTYPE) {
        cudaMalloc((void**)&database::diffGPU, sizeof(ValueType)*database::cnn->GetRoot()->GetNumEl());
    }
    database::deriv = new ValueType[database::cnn->GetRoot()->GetNumEl()];
    std::fill(database::deriv, database::deriv + database::cnn->GetRoot()->GetNumEl(), ValueType(0.0));
}

pair<double,double> APSVMTrain(string mode){
    int ClusterSize = 0;
    int ClusterID = 0;
#ifdef WITH_MPI
    ClusterSize = MPI::COMM_WORLD.Get_size();
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif
    assert(Conf::pos_num + Conf::neg_num == imdb::imnum);
    int pSize = Conf::pos_num;
    int nSize = Conf::neg_num;
    ValueType* scores = new ValueType[ClusterSize * Conf::batch_size]();

    std::memset(database::data, 0, sizeof(database::data));
    for(int i = Conf::batch_size * ClusterID, e = std::min(Conf::batch_size*(ClusterID+1), pSize+nSize); i<e; i++){
        std::bernoulli_distribution do_mirror(0.5);
        Image::crop(i, database::data + (i - Conf::batch_size*ClusterID)*Conf::crop_size*Conf::crop_size*3,
                    do_mirror(LSDN::Instance().cpu_generator()));
    }
    if (GPUTYPE) {
        cudaMemcpy((char*)database::dataGPU, (char*)database::data, sizeof(ValueType)*Conf::crop_size*
                   Conf::crop_size * 3 * Conf::batch_size, cudaMemcpyHostToDevice);
    }
    database::cnn->ForwardPass(TRAIN);
    NodeType* rootNodeDeepNet = database::cnn->GetRoot();
    if (GPUTYPE) {
        cudaMemcpy((char*)database::netOutput, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    }
    for(int id = 0; id < Conf::batch_size; id++){
        scores[id] = database::netOutput[id];
    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Gather(MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, scores, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Gather(scores, Conf::batch_size, MPI::FLOAT, nullptr, Conf::batch_size, MPI::FLOAT, 0);
#endif
    ValueType *derivs = new ValueType[ClusterSize * Conf::batch_size]();
    double ap = 0;
    double ap_clear = 0;
    if(ClusterID == 0){
        vector<ValueType> scorep, scoren;
        vector<ValueType> scorep_clear, scoren_clear;
        vector<int> posp, posn;
        for(int i = 0; i < pSize + nSize; i++){
            if(imdb::rel[i]){
                scorep.push_back(scores[i]);
                posp.push_back(i);
            }
            else{
                scoren.push_back(scores[i]);
                posn.push_back(i);
            }
            if(imdb::rel_clear[i])
                scorep_clear.push_back(scores[i]);
            else
                scoren_clear.push_back(scores[i]);
        }
        vector<ValueType> combineScores(scorep);
        for(auto x:scoren) combineScores.push_back(x);

        vector<ValueType> combineScores_clear(scorep_clear);
        for(auto x:scoren_clear) combineScores_clear.push_back(x);

        vector<size_t> iotaW_clear(pSize+nSize,0),iotaW(pSize+nSize,0), iotaDirect(pSize+nSize,0);
        for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaW_clear[i]=iotaW[i] = iotaDirect[i] = i;

        std::sort(iotaW.begin(),iotaW.end(),[&](int a,int b){return (combineScores[a] > combineScores[b]) ? true:false;});
        std::sort(iotaW_clear.begin(), iotaW_clear.end(), [&](int a, int b){
            return combineScores_clear[a] > combineScores_clear[b];
        });

        vector<size_t> posW(pSize + nSize,0), posDirect(pSize + nSize,0);
        for(size_t i = 0, e = iotaW.size(); i != e; i++)    posW[iotaW[i]]=i;

        std::sort(iotaDirect.begin(), iotaDirect.begin()+pSize, [&](int a,int b){
            return (combineScores[a] > combineScores[b]) ? true:false;
        });

        std::sort(iotaDirect.begin()+pSize, iotaDirect.end(), [&](int a,int b){
            return (combineScores[a] > combineScores[b]) ? true:false;
        });

        vector<double> pScores(pSize,0), nScores(nSize,0);
        for(size_t i = 0; i != pSize; i++)  pScores[i] = combineScores[iotaDirect[i]];
        for(size_t i = 0; i != nSize; i++)  nScores[i] = combineScores[iotaDirect[i+pSize]];
        vector<size_t> posTmp(pSize+nSize,0);
        DP<true>(pScores,nScores,posTmp,1);
        for(size_t i = 0, e = posTmp.size(); i != e; i++)
            posDirect[iotaDirect[i]] = posTmp[i];

        // Get the derivatives
        for(int i = 0; i != pSize; i++){
            double sum = 0;
            for(int j = 0; j != nSize; j++){
                int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
                int yt = 1;
                sum += yDirect - yt;
            }
            sum /= (pSize * nSize);
            derivs[posp[i]] = sum;
        }

        for(int j = 0; j != nSize; j++){
            double sum = 0;
            for(int i = 0; i != pSize; i++){
                int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
                int yt = 1;
                sum -= yDirect - yt;
            }
            sum /= (pSize * nSize);
            derivs[posn[j]] = sum;
        }

        int count = 0;
        for(int i = 0; i < pSize + nSize; i++)
            if(iotaW[i] < pSize){
                count++;
                ap += count / double(i + 1);
            }
        ap /= count;

        count = 0;
        for(int i = 0; i < pSize + nSize; i++)
            if(iotaW_clear[i] < pSize){
                count++;
                ap_clear += count / double(i+1);
            }
        ap_clear /= count;

    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Scatter(derivs, Conf::batch_size, MPI::FLOAT, MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Scatter(nullptr, Conf::batch_size, MPI::FLOAT, derivs, Conf::batch_size, MPI::FLOAT, 0);
#endif
    std::fill(database::deriv, database::deriv + database::cnn->GetRoot()->GetNumEl(), ValueType(0.0));
    for(int id = 0; id < Conf::batch_size; id++){
        database::deriv[id] = derivs[id];
    }
    ValueType** diff = database::cnn->GetRoot()->GetDiffGradientAndEmpMean();
    //cout << "Dimension of gradient: " << rootNodeDeepNet->GetNumEl() << endl;
    ValueType* deriv = database::deriv, *diffGPU = database::diffGPU;
    if (GPUTYPE) {
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    if (GPUTYPE){
        cudaMemcpy(diffGPU, deriv, sizeof(ValueType) * database::cnn->GetRoot()->GetNumEl(), cudaMemcpyHostToDevice);
    }
    database::cnn->BackwardPass();

    if(ClusterID == 0){
        vector<ValueType> grad;
        database::cnnParams.GetDerivative(i2t<GPUTYPE>(), grad);
        double norm = 0;
        for(auto x : grad)	norm += x * x;
        norm = std::sqrt(norm);
        cout<<"L2 norm of gradient: " << norm << endl;
    }
    database::cnnParams.Update(ClusterSize);
    database::cnnParams.ResetGradient(i2t<GPUTYPE>());
   // primal += DeepNet16Params.GetRegularization();

    delete[] scores;
    delete[] derivs;

    return make_pair(ap,ap_clear);
}

template<bool positive=true>
pair<double,double> APDLMTrain(string mode, const double epsilon){
    int ClusterSize = 0;
    int ClusterID = 0;
#ifdef WITH_MPI
    ClusterSize = MPI::COMM_WORLD.Get_size();
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif
    assert(Conf::pos_num + Conf::neg_num == imdb::imnum);
    int pSize = Conf::pos_num;
    int nSize = Conf::neg_num;
    ValueType* scores = new ValueType[ClusterSize * Conf::batch_size]();

    std::memset(database::data, 0, sizeof(database::data));
    for(int i = Conf::batch_size * ClusterID, e = std::min(Conf::batch_size*(ClusterID+1), pSize+nSize); i<e; i++){
        std::bernoulli_distribution do_mirror(0.5);
        Image::crop(i, database::data + (i - Conf::batch_size*ClusterID)*Conf::crop_size*Conf::crop_size*3,
                    do_mirror(LSDN::Instance().cpu_generator()));
    }
    if (GPUTYPE) {
        cudaMemcpy((char*)database::dataGPU, (char*)database::data, sizeof(ValueType)*Conf::crop_size*
                   Conf::crop_size * 3 * Conf::batch_size, cudaMemcpyHostToDevice);
    }
    database::cnn->ForwardPass(TRAIN);
    NodeType* rootNodeDeepNet = database::cnn->GetRoot();
    if (GPUTYPE) {
        cudaMemcpy((char*)database::netOutput, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    }
    for(int id = 0; id < Conf::batch_size; id++){
        scores[id] = database::netOutput[id];
    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Gather(MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, scores, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Gather(scores, Conf::batch_size, MPI::FLOAT, nullptr, Conf::batch_size, MPI::FLOAT, 0);
#endif
    ValueType *derivs = new ValueType[ClusterSize * Conf::batch_size]();
    double ap = 0;
    double ap_clear = 0;
    if(ClusterID == 0){
        vector<ValueType> scorep_clear, scoren_clear;
        vector<ValueType> scorep, scoren;
        vector<int> posp, posn;
        for(int i = 0; i < pSize + nSize; i++){
            if(imdb::rel[i]){
                scorep.push_back(scores[i]);
                posp.push_back(i);
            }
            else{
                scoren.push_back(scores[i]);
                posn.push_back(i);
            }
            if(imdb::rel_clear[i])
                scorep_clear.push_back(scores[i]);
            else
                scoren_clear.push_back(scores[i]);
        }
        vector<ValueType> combineScores_clear(scorep_clear);
        for(auto x:scoren_clear) combineScores_clear.push_back(x);
        vector<ValueType> combineScores(scorep);
        for(auto x:scoren) combineScores.push_back(x);

        vector<size_t> iotaW(pSize+nSize,0), iotaDirect(pSize+nSize,0), iotaW_clear(pSize+nSize,0);
        for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaW[i] = iotaDirect[i] = iotaW_clear[i] = i;

        std::sort(iotaW.begin(),iotaW.end(),[&](int a,int b){return (combineScores[a] > combineScores[b]) ? true:false;});
        std::sort(iotaW_clear.begin(), iotaW_clear.end(), [&](int a, int b){
            return combineScores_clear[a] > combineScores_clear[b];
        });


        vector<size_t> posW(pSize + nSize,0), posDirect(pSize + nSize,0);
        for(size_t i = 0, e = iotaW.size(); i != e; i++)    posW[iotaW[i]]=i;

        std::sort(iotaDirect.begin(), iotaDirect.begin()+pSize, [&](int a,int b){
            return (combineScores[a] > combineScores[b]) ? true:false;
        });

        std::sort(iotaDirect.begin()+pSize, iotaDirect.end(), [&](int a,int b){
            return (combineScores[a] > combineScores[b]) ? true:false;
        });

        vector<double> pScores(pSize,0), nScores(nSize,0);
        for(size_t i = 0; i != pSize; i++)  pScores[i] = combineScores[iotaDirect[i]];
        for(size_t i = 0; i != nSize; i++)  nScores[i] = combineScores[iotaDirect[i+pSize]];
        vector<size_t> posTmp(pSize+nSize,0);
        DP<positive>(pScores,nScores,posTmp,epsilon);
        for(size_t i = 0, e = posTmp.size(); i != e; i++)
            posDirect[iotaDirect[i]] = posTmp[i];

        // Get the derivatives
        for(int i = 0; i != pSize; i++){
            double sum = 0;
            for(int j = 0; j != nSize; j++){
                int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
                int yW = (posW[i] < posW[j + pSize]) ? 1 : -1;
                if(positive) sum += yDirect - yW;
                else sum += yW - yDirect;
            }
            sum /= (epsilon * pSize * nSize);
            derivs[posp[i]] = sum;
        }

        for(int j = 0; j != nSize; j++){
            double sum = 0;
            for(int i = 0; i != pSize; i++){
                int yDirect = (posDirect[i] < posDirect[j + pSize]) ? 1 : -1;
                int yW = (posW[i] < posW[j + pSize]) ? 1 : -1;
                if(positive) sum -= yDirect - yW;
                else sum -= yW - yDirect;
            }
            sum /= (epsilon * pSize * nSize);
            derivs[posn[j]] = sum;
        }

        int count = 0;
        for(int i = 0; i < pSize + nSize; i++)
            if(iotaW[i] < pSize){
                count++;
                ap += count / double(i + 1);
            }
        ap /= count;

        count = 0;
        for(int i = 0; i < pSize + nSize; i++)
            if(iotaW_clear[i] < pSize){
                count++;
                ap_clear += count / double(i+1);
            }
        ap_clear /= count;

    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Scatter(derivs, Conf::batch_size, MPI::FLOAT, MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Scatter(nullptr, Conf::batch_size, MPI::FLOAT, derivs, Conf::batch_size, MPI::FLOAT, 0);
#endif
    std::fill(database::deriv, database::deriv + database::cnn->GetRoot()->GetNumEl(), ValueType(0.0));
    for(int id = 0; id < Conf::batch_size; id++){
        database::deriv[id] = derivs[id];
    }
    ValueType** diff = database::cnn->GetRoot()->GetDiffGradientAndEmpMean();
    //cout << "Dimension of gradient: " << rootNodeDeepNet->GetNumEl() << endl;
    ValueType* deriv = database::deriv, *diffGPU = database::diffGPU;
    if (GPUTYPE) {
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    if (GPUTYPE){
        cudaMemcpy(diffGPU, deriv, sizeof(ValueType) * database::cnn->GetRoot()->GetNumEl(), cudaMemcpyHostToDevice);
    }
    database::cnn->BackwardPass();

    if(ClusterID == 0){
        vector<ValueType> grad;
        database::cnnParams.GetDerivative(i2t<GPUTYPE>(), grad);
        double norm = 0;
        for(auto x : grad)	norm += x * x;
        norm = std::sqrt(norm);
        cout<<"L2 norm of gradient: " << norm << endl;
    }
    database::cnnParams.Update(ClusterSize);
    database::cnnParams.ResetGradient(i2t<GPUTYPE>());
   // primal += DeepNet16Params.GetRegularization();

    delete[] scores;
    delete[] derivs;

    return make_pair(ap,ap_clear);
}

template<bool positive=true>
void train(string mode, const double epsilon){
    int ClusterID = 0;
#ifdef WITH_MPI
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif

    std::queue<double> qsum10;
    std::queue<double> qsum100;

    double runsum10 = 0, runsum100 = 0;

    for(int iter= 0; iter < Conf::max_iter; iter++){
        auto AP = APDLMTrain<positive>(mode, epsilon);
        if(ClusterID == 0){
            cout<<"Iteration: " << iter <<endl;
            cout<<"AP on training batch with outliers: " << AP.first << endl;
            cout<<"AP on training batch without outliers: " << AP.second << endl;

            if(iter < 10){
                qsum10.push(AP.second);
                runsum10 += AP.second;
            }
            else{
                qsum10.push(AP.second);
                runsum10 -= qsum10.front();
                runsum10 += AP.second;
                qsum10.pop();
                cout<<" running AP on 10 batches without outliers: " << runsum10 / 10 << endl;
            }
            if(iter < 100){
                runsum100 += AP.second;
                qsum100.push(AP.second);
            }
            else{
                runsum100 += AP.second;
                qsum100.push(AP.second);
                runsum100 -= qsum100.front();
                qsum100.pop();
                cout << "running AP on 100 batches without outliers: " << runsum100 / 100 << endl;
            }

            if((iter+1) % Conf::snap_shot == 0){
                char name[100]={0};
                std::sprintf(name,Conf::save_weight.c_str(), iter+1);
                database::savename = Conf::dir_weights + name;
                database::saveWeights();
            }
        }
        if((iter + 1)% Conf::step == 0)
            database::cnnParams.ReduceStepSize();
    }
}

void test(string testset){
    string filename = Conf::dir_res + Conf::res_name;

    std::ofstream fout(filename.c_str());

    int batch_num = imdb::imnum / Conf::batch_size + 1;
    vector<ValueType> scores(imdb::imnum,0);

    for(int batch = 0; batch < batch_num; batch++){
          // Load pictures
        std::memset(database::data, 0, sizeof(database::data)) ;
        for(int k = batch * Conf::batch_size, e = std::min((batch + 1) * Conf::batch_size, imdb::imnum); k < e; k++){
            int rk = k - batch * Conf::batch_size;
            Image::crop(k, database::data + rk * Conf::crop_size * Conf::crop_size * 3, false);
        }

          if (GPUTYPE) {
              cudaMemcpy((char*)database::dataGPU, (char*)database::data, sizeof(ValueType)*Conf::crop_size*
                         Conf::crop_size * 3 * Conf::batch_size, cudaMemcpyHostToDevice);
          }

          database::cnn->ForwardPass(TEST);
          cout<<"For batch "<<batch<<endl;

          NodeType* rootNodeDeepNet = database::cnn->GetRoot();
          if (GPUTYPE) {
              cudaMemcpy((char*)database::netOutput, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
          }

          for(int k = batch * Conf::batch_size, e = std::min((batch + 1) * Conf::batch_size, imdb::imnum); k < e; k++)
              scores[k] = database::netOutput[k - batch * Conf::batch_size];
    }

    int pSize = Conf::pos_num;
    int nSize = Conf::neg_num;
    vector<ValueType> scorep, scoren;
    vector<int> posp, posn;
    for(int i = 0; i < pSize + nSize; i++){
        if(imdb::rel[i]){
            scorep.push_back(scores[i]);
            posp.push_back(i);
        }
        else{
            scoren.push_back(scores[i]);
            posn.push_back(i);
        }
    }
    vector<ValueType> combineScores(scorep);
    for(auto x:scoren) combineScores.push_back(x);

    vector<size_t> iotaW(pSize+nSize,0);
    for(size_t i = 0, e = pSize+nSize; i != e; i++) iotaW[i] = i;

    std::sort(iotaW.begin(),iotaW.end(),[&](int a,int b){return (combineScores[a] > combineScores[b]) ? true:false;});
    int count = 0;
    double ap = 0;
    for(int i = 0; i < pSize + nSize; i++)
        if(iotaW[i] < pSize){
            count++;
            ap += count / double(i + 1);
        }
    ap /= count;
    fout<<ap<<endl;
    cout<<"AP on " << testset << " is " << ap << endl;
/*
    for(int i = 0; i < raw_scores.size(); i++)
        fout<<imdb::ids[i]<<" "<<imdb::objind[i]<<" "<<raw_scores[i]<<endl;
*/
}
double HingeTrain(string mode){
    int ClusterSize = 0;
    int ClusterID = 0;
#ifdef WITH_MPI
    ClusterSize = MPI::COMM_WORLD.Get_size();
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif
    assert(Conf::pos_num + Conf::neg_num == imdb::imnum);
    ValueType* scores = new ValueType[ClusterSize * Conf::batch_size]();

    std::memset(database::data, 0, sizeof(database::data));
    for(int i = Conf::batch_size * ClusterID, e = std::min(Conf::batch_size*(ClusterID+1), imdb::imnum); i<e; i++){
        std::bernoulli_distribution do_mirror(0.5);
        Image::crop(i, database::data + (i - Conf::batch_size*ClusterID)*Conf::crop_size*Conf::crop_size*3,
                    do_mirror(LSDN::Instance().cpu_generator()));
    }
    if (GPUTYPE) {
        cudaMemcpy((char*)database::dataGPU, (char*)database::data, sizeof(ValueType)*Conf::crop_size*
                   Conf::crop_size * 3 * Conf::batch_size, cudaMemcpyHostToDevice);
    }
    database::cnn->ForwardPass(TRAIN);
    NodeType* rootNodeDeepNet = database::cnn->GetRoot();
    if (GPUTYPE) {
        cudaMemcpy((char*)database::netOutput, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    }
    for(int id = 0; id < Conf::batch_size; id++){
        scores[id] = database::netOutput[id];
    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Gather(MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, scores, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Gather(scores, Conf::batch_size, MPI::FLOAT, nullptr, Conf::batch_size, MPI::FLOAT, 0);
#endif
    ValueType *derivs = new ValueType[ClusterSize * Conf::batch_size]();
    double acc = 0;
    if(ClusterID == 0){
        for(int i = 0; i < imdb::imnum; i++){

            if(imdb::rel[i]){
                if(scores[i] < 1) derivs[i] = -1;
                else derivs[i] = 0;
            }
            else if(!imdb::rel[i]){
                if(scores[i] > -1) derivs[i] = 1;
                else derivs[i] = 0;
            }

            if(scores[i] >= 0 && imdb::rel[i])
                acc++;
            else if(scores[i] < 0 && !imdb::rel[i])
                acc++;

        }
        acc /= imdb::imnum;
    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Scatter(derivs, Conf::batch_size, MPI::FLOAT, MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Scatter(nullptr, Conf::batch_size, MPI::FLOAT, derivs, Conf::batch_size, MPI::FLOAT, 0);
#endif
    std::fill(database::deriv, database::deriv + database::cnn->GetRoot()->GetNumEl(), ValueType(0.0));
    for(int id = 0; id < Conf::batch_size; id++){
        database::deriv[id] = derivs[id];
    }
    ValueType** diff = database::cnn->GetRoot()->GetDiffGradientAndEmpMean();
    //cout << "Dimension of gradient: " << rootNodeDeepNet->GetNumEl() << endl;
    ValueType* deriv = database::deriv, *diffGPU = database::diffGPU;
    if (GPUTYPE) {
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    if (GPUTYPE){
        cudaMemcpy(diffGPU, deriv, sizeof(ValueType) * database::cnn->GetRoot()->GetNumEl(), cudaMemcpyHostToDevice);
    }
    database::cnn->BackwardPass();

    if(ClusterID == 0){
        vector<ValueType> grad;
        database::cnnParams.GetDerivative(i2t<GPUTYPE>(), grad);
        double norm = 0;
        for(auto x : grad)	norm += x * x;
        norm = std::sqrt(norm);
        cout<<"L2 norm of gradient: " << norm << endl;
    }
    database::cnnParams.Update(ClusterSize);
    database::cnnParams.ResetGradient(i2t<GPUTYPE>());
   // primal += DeepNet16Params.GetRegularization();

    delete[] scores;
    delete[] derivs;

    return acc;
}

double LogisticTrain(string mode){
    int ClusterSize = 0;
    int ClusterID = 0;
#ifdef WITH_MPI
    ClusterSize = MPI::COMM_WORLD.Get_size();
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif
    assert(Conf::pos_num + Conf::neg_num == imdb::imnum);
    ValueType* scores = new ValueType[ClusterSize * Conf::batch_size]();

    std::memset(database::data, 0, sizeof(database::data));
    for(int i = Conf::batch_size * ClusterID, e = std::min(Conf::batch_size*(ClusterID+1), imdb::imnum); i<e; i++){
        std::bernoulli_distribution do_mirror(0.5);
        Image::crop(i, database::data + (i - Conf::batch_size*ClusterID)*Conf::crop_size*Conf::crop_size*3,
                    do_mirror(LSDN::Instance().cpu_generator()));
    }
    if (GPUTYPE) {
        cudaMemcpy((char*)database::dataGPU, (char*)database::data, sizeof(ValueType)*Conf::crop_size*
                   Conf::crop_size * 3 * Conf::batch_size, cudaMemcpyHostToDevice);
    }
    database::cnn->ForwardPass(TRAIN);
    NodeType* rootNodeDeepNet = database::cnn->GetRoot();
    if (GPUTYPE) {
        cudaMemcpy((char*)database::netOutput, rootNodeDeepNet->GetValuePtr(), rootNodeDeepNet->GetNumEl()*sizeof(ValueType), cudaMemcpyDeviceToHost);
    }
    for(int id = 0; id < Conf::batch_size; id++){
        scores[id] = database::netOutput[id];
    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Gather(MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, scores, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Gather(scores, Conf::batch_size, MPI::FLOAT, nullptr, Conf::batch_size, MPI::FLOAT, 0);
#endif
    ValueType *derivs = new ValueType[ClusterSize * Conf::batch_size]();
    double acc = 0;
    if(ClusterID == 0){
        for(int i = 0; i < imdb::imnum; i++){
            int t = imdb::rel[i] ? 1 : 0;
            double p = 1.0 / (1.0 + std::exp(-scores[i]));
            if(p >= 0.5 && imdb::rel[i])
                acc++;
            else if(p < 0.5 && !imdb::rel[i])
                acc++;
            derivs[i] = p - t;
        }
        acc /= imdb::imnum;
    }
#ifdef WITH_MPI
    if(ClusterID == 0)
        MPI::COMM_WORLD.Scatter(derivs, Conf::batch_size, MPI::FLOAT, MPI::IN_PLACE, Conf::batch_size, MPI::FLOAT, 0);
    else
        MPI::COMM_WORLD.Scatter(nullptr, Conf::batch_size, MPI::FLOAT, derivs, Conf::batch_size, MPI::FLOAT, 0);
#endif
    std::fill(database::deriv, database::deriv + database::cnn->GetRoot()->GetNumEl(), ValueType(0.0));
    for(int id = 0; id < Conf::batch_size; id++){
        database::deriv[id] = derivs[id];
    }
    ValueType** diff = database::cnn->GetRoot()->GetDiffGradientAndEmpMean();
    //cout << "Dimension of gradient: " << rootNodeDeepNet->GetNumEl() << endl;
    ValueType* deriv = database::deriv, *diffGPU = database::diffGPU;
    if (GPUTYPE) {
        *diff = diffGPU;
    } else {
        *diff = deriv;
    }

    if (GPUTYPE){
        cudaMemcpy(diffGPU, deriv, sizeof(ValueType) * database::cnn->GetRoot()->GetNumEl(), cudaMemcpyHostToDevice);
    }
    database::cnn->BackwardPass();

    if(ClusterID == 0){
        vector<ValueType> grad;
        database::cnnParams.GetDerivative(i2t<GPUTYPE>(), grad);
        double norm = 0;
        for(auto x : grad)	norm += x * x;
        norm = std::sqrt(norm);
        cout<<"L2 norm of gradient: " << norm << endl;
    }
    database::cnnParams.Update(ClusterSize);
    database::cnnParams.ResetGradient(i2t<GPUTYPE>());
   // primal += DeepNet16Params.GetRegularization();

    delete[] scores;
    delete[] derivs;

    return acc;
}

template pair<double,double> APDLMTrain<true>(string, const double epsilon);
template pair<double,double> APDLMTrain<false>(string, const double epsilon);
template void train<true>(string, const double epsilon);
template void train<false>(string, const double epsilon);
