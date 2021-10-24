# Training Deep Neural Networks via Direct Loss Minimization

This is the code for our ICML'16 paper:

* Yang Song, Alexander G. Schwing, Richard S. Zemel, Raquel Urtasun, [*Training Deep Neural Networks via Direct Loss Minimization*](http://jmlr.org/proceedings/papers/v48/songb16.pdf), International Conference on Machine Learning, New York City, USA, 2016.

Please cite the above paper if you use our code. For simplicity, we only include the code for action classification. We basically use the same code for object detection with minor modifications.

### Dependencies
Our codes depend on **qmake**, **CUDA**, **MPICH**, **CImg**, **LibLSDN** (see the following section) and some MATLAB codes from [**PASCAL VOC2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) development toolkits. It is compulsory to have these downloaded (and installed).

The main source code was written in **C++11**.
### LibLSDN
Our code is based on a deep learning library _LibLSDN_, which is written and maintained by [Alexander G. Schwing](http://www.alexander-schwing.de/). We include part of that library in our repository, including `CPrecisionTimer.h`, `LSDN_Common.h`, `LSDN_CudaCommon.h` and all files in `LibLSDN/`.

In order to install this library, please first modify the paths accordingly in `Makefile` and then run

```bash
make LibLSDN.a
```

### Data
We use the [**PASCAL VOC2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) dataset, which can be downloaded at the website. The dataset contains a `VOCcode/` folder, which contains a lot of utility MATLAB codes helpful for preprocessing data. 

After extracting VOC2012 dataset, you should put [`preprocessing/save_action_split.m`](preprocessing/save_action_split.m) under the `VOCdevkit2012/` folder and run it to get preprocessed data files. They have the format of `*_info.dat` and `*_ims.dat`.

### Training and Validation / Testing
Previous to running the program, please modify the paths and parameters in `src/APDLM.pro` and `src/Conf.cpp` according to your own settings.

First run the following commands

```
qmake
make
```
to compile the program.

Then run

```bash
mpiexec -n 2 ./APDLM [alpha] [beta] [epsilon] [class_name] [gpu1] [gpu2]
```
Here `alpha` is the learning rate, `beta` is the L2 regularization coefficient, `epsilon` is the sharpness parameter as defined in the paper, `class_name` is the name of the class, `gpu1` and `gpu2` are indices of GPUs. You can also change the number of GPUs.

### Notes

The AlexNet weight `AlexNetWeightsSingleDimensionOutput.dat` used for finetuning is available at this [Google drive](https://drive.google.com/file/d/1wRiDvwVJGR00ZwaWSW5MvavZIxfi81Dl/view?usp=sharing).
