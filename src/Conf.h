#ifndef CONF_HEADER
#define CONF_HEADER
#include <string>
#include <random>
using std::string;

#define GPUTYPE true
#define cimg_use_jpeg
#define WITH_MPI
#ifdef WITH_MPI
    #include "mpi.h"
#endif

typedef float ValueType;
typedef int SizeType;

class Conf{
public:
    static string DATASET;
    static string RUNNINGPATH;
    static string dir_mats;
    static string dir_weights;
    static int batch_size;
    static int crop_size, crop_padding;
    static int pos_num;
    static int neg_num;
    static int snap_shot;
    static int step;
    static string dir_res;
    static string res_name;
    static string load_weight;
    static string save_weight;
    static string mode;
    static double outlier_rate;
    static int max_iter;
};
#endif // CONF

