/* Code written by Yang Song (yangsong@cs.stanford.edu)
 * Utility functions for loading and initializing images, labels and weights.
*/
#include "database.h"
#include "Image.h"
#include "Conf.h"
#include "cnn.h"
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <utility>
#include <stdexcept>
#include <cassert>
#include <random>
#include "CImg.h"
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

vector<string> imdb::ids;
vector<long long> imdb::pos;
vector<int> imdb::objind;
vector<bool> imdb::rel; // whether a image is relevant
vector<bool> imdb::rel_clear;
char* imdb::ims;
int imdb::imnum = 0;
std::random_device imdb::rd;
std::mt19937 imdb::gen(imdb::rd());
std::uniform_real_distribution<double> imdb::uniform(0,1);

// Load label information
void imdb::load(string mode){
    string fullpath = Conf::dir_mats + Conf::DATASET+"_action_" + mode + "_"+database::iclass+"_info.dat";
    ifstream fin(fullpath);
    assert(fin);
    string imname;
    int objid;
    int N = 0;
    int label = 0;
    Conf::pos_num = 0;
    Conf::neg_num = 0;
    imdb::objind.clear();
    imdb::ids.clear();
    imdb::rel.clear();

    // Read the preprocessed information
    while(fin >> imname >> objid >> label){
        imdb::objind.push_back(objid);
        imdb::ids.push_back(imname);
        imdb::rel_clear.push_back(label == 1);
        if(imdb::uniform(imdb::gen) <= Conf::outlier_rate)
            imdb::rel.push_back(label != 1);
        else
            imdb::rel.push_back(label == 1);

        if(imdb::rel.back()){
            Conf::pos_num ++;
            database::pos.push_back(N);
        }
        else{
            Conf::neg_num ++;
            database::neg.push_back(N);
        }

        N++;
    }
    imdb::imnum = N;
    cout<<"Outlier rate: " << Conf::outlier_rate << endl;
    cout<<"Total number of positives: " << Conf::pos_num <<endl;
    cout<<"Total number of negatives: " << Conf::neg_num << endl;
    cout<<"Total number of images: " << imdb::imnum <<endl;
}

vector<int> database::pos, database::neg;
// Load the preprocessed image files
void imdb::loadims(string mode){
    if(ims != nullptr)  delete[] ims;

    ifstream fin(Conf::dir_mats + Conf::DATASET+"_action_" + mode + "_"+database::iclass+"_ims.dat", std::ios::binary);
    assert(fin);

    cout<<"Loading " << mode << " images."<<endl;
    imdb::ims = new char[Conf::crop_size * Conf::crop_size * 3 * imdb::imnum];
    fin.read((char*)imdb::ims, sizeof(char) * Conf::crop_size * Conf::crop_size * 3 * imdb::imnum);
}

CompTree *database::cnn = new CompTree;
ParameterContainer<ValueType,SizeType,GPUTYPE,false> database::cnnParams;
DataType *database::cnnData = new DataType(DataType::NodeParameters());
string database::loadname;
string database::savename;
string database::iclass;
ValueType * database::data = nullptr;
ValueType * database::dataGPU = nullptr;
ValueType* database::netOutput = nullptr;
ValueType* database::deriv = nullptr;
ValueType* database::diffGPU = nullptr;

void database::loadWeights(){
    SetWeights(loadname,cnnParams);
}
void database::saveWeights(){
    vector<ValueType>* result = cnnParams.GetWeights(i2t<GPUTYPE>());
    std::ofstream ofs(savename, std::ios_base::binary | std::ios_base::out);
    ofs.write((char*)&(*result)[0], result->size()*sizeof(ValueType));
}
void database::init(const double alpha, const double beta, string classname, int GPUid){
    cout << "Loading image information..." << endl;
    Conf::batch_size = std::ceil((Conf::pos_num + Conf::neg_num) / double(MPI::COMM_WORLD.Get_size()));
    cout<<"Size of batch: "<<Conf::batch_size<<endl;
    initCNN(alpha,beta, GPUid);
}
