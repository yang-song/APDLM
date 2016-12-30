/* Code written by Yang Song (yangsong@cs.stanford.edu)
 * This is the code for VOC2012 action classification.
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <sstream>
#include "dp.h"
#include "cnn.h"
#include "Conf.h"
#include "database.h"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char *argv[])
{
    int ClusterSize = 1;
    int ClusterID = 0;
#ifdef WITH_MPI
    if (!MPI::Is_initialized()) {
        MPI::Init();
    }
    ClusterSize = MPI::COMM_WORLD.Get_size();
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif   
    /* The program takes 4 arguments:
     * alpha: learning rate
     * beta: L2 regularization coefficient
     * epsilon: as defined in paper
     * to control the sharpness of direct loss minimization
     * whichClass: a string representing which class we are considering
    */
    double alpha, beta, epsilon;
    string whichClass;
    alpha = std::stod(argv[1]);
    beta = std::stod(argv[2]);
    epsilon = std::stod(argv[3]);
    whichClass = argv[4];
    database::iclass = whichClass;

    // Generate the name of log files
    std::stringstream format("");
    format<<"action-"<<Conf::mode<<"-"<<Conf::outlier_rate<<"-"<<alpha<<"-"<<beta<<"-"<<epsilon<<"-"<<whichClass<<".txt";
    format>>Conf::res_name;

    format.clear();
    format.str("");
    format<<"action-"<<Conf::mode<<"-"<<Conf::outlier_rate<<"-"<<alpha<<"-"<<beta<<"-"<<epsilon<<"-"<<whichClass<<"-iter%d.dat";
    format>>Conf::save_weight;
    if(ClusterID == 0){
        cout<<"res_name: "<<Conf::res_name<<endl;
        cout<<"save_weight: "<<Conf::save_weight<<endl;
    }
    // load the training dataset
    imdb::load("train");
    imdb::loadims("train");
    database::init(alpha,beta,whichClass,std::atoi(argv[5+ClusterID]));// Set GPUID here!
    cout<<"after init!"<<endl;
    train("train",epsilon);

    /////////////////
    // load and test on val or test, depending on Conf::mode
    if(ClusterID == 0){
        Conf::outlier_rate = -1;
        imdb::load(Conf::mode);
        imdb::loadims(Conf::mode);
        test(Conf::mode);
    }
#ifdef WITH_MPI
    if (!MPI::Is_finalized()) {
        MPI::Finalize();
    }
#endif
    return 0;
}
