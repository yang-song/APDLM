#ifndef DATABASE_HEADER
#define DATABASE_HEADER

#include<string>
#include<vector>
#include<set>
#include<map>
#include<stack>
#include<utility>
#include<string>
#include "Conf.h"
#include "cnn.h"

using std::set;
using std::map;
using std::vector;
using std::string;
using std::stack;
using std::pair;
using std::make_pair;
using std::string;

class imdb{
public:
    static void load(string mode);
    static vector<string> ids;
    static vector<int> objind;
    static vector<bool> rel;
    static vector<bool> rel_clear;
    static char* ims;
    static int imnum;
    static vector<long long> pos;
    static void load();
    static void loadims(string mode);
private:
    static std::random_device rd;
    static std::mt19937 gen;
    static std::uniform_real_distribution<double> uniform;
};

class database{
public:
    static CompTree *cnn;
    static ParameterContainer<ValueType,SizeType,GPUTYPE,false> cnnParams;
    static DataType *cnnData;
    static string loadname;
    static string savename;
    static string iclass;
    static ValueType* data;
    static ValueType* dataGPU;
    static ValueType* netOutput;
    static ValueType* deriv;
    static ValueType* diffGPU;

    static vector<int> pos, neg;

    static void deleteCNN();
    static void init(const double alpha, const double beta, string classname, int GPUid);
    static void loadWeights();
    static void saveWeights();
};
#endif // INFO

