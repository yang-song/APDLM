/* Code written by Yang Song (yangsong@cs.stanford.edu)
 * Configuration file.
 * Please modify all paths according to your settings.
*/
#include "Conf.h"

string Conf::DATASET("VOC2012");
string Conf::RUNNINGPATH("/ais/gobi3/u/yangsong/dataset/PASCAL"); // The directory of dataset
string Conf::dir_mats = Conf::RUNNINGPATH + "/" + Conf::DATASET + "/random_action/"; // The directory of preprocessed image files
string Conf::dir_weights = "/u/yangsong/largeStorage/pascalExp/weights/"; // The directory of weight files

string Conf::dir_res = "/u/yangsong/largeStorage/pascalExp/results/"; // The directory of result logs

string Conf::res_name = "mpi_val.txt"; // The name of log file
string Conf::load_weight = "AlexNetWeightsSingleDimensionOutput.dat"; // The name of the weight to load.
                                                                      // Please contact Yang Song for this intialization weight file.
string Conf::save_weight = "mpi_weight%d.bak"; // The name format of cached weights
string Conf::mode = "val"; // Test on which dataset, "val" or "test"
double Conf::outlier_rate = 0.1; // The flip ratio in paper
int Conf::batch_size = 0; // Do not need to specify in action classification, as
                          // it would be calculated automatically, depending on the number of available GPUs.
int Conf::pos_num = 0; // This will be calculated automatically in action classification
int Conf::neg_num = 0; // This will also be calculated automatically
int Conf::snap_shot = 300; // How often do we cache weights
int Conf::step = 20000; // How often do we halve the step size
int Conf::crop_size = 227; // Crop the original imnage to crop_size * crop_size
int Conf::crop_padding = 16; // The padding as in RCNN
