TARGET = APDLM
CONFIG += c++11
# The path to compiled library LibLSDN
LIBS += -L../ -lLSDN
# The root path of CUDA
LIBS += -L/ais/gobi3/pkgs/cuda-7.0/lib64 -lcurand -lcublas -lcudart -lblas -lm
# The root path of MPICH
LIBS += -L/pkgs/mpich-3.0.4/lib/x86_64-linux-gnu -lmpich -lmpichcxx
INCLUDEPATH += /pkgs/mpich-3.0.4/include

LIBS += -lX11 -ljpeg

HEADERS += \
    dp.h \
    CImg.h \
    Conf.h \
    Image.h \
    database.h \
    cnn.h

SOURCES += \
    dp.cpp \
    main.cpp \
    Conf.cpp \
    Image.cpp \
    database.cpp \
    cnn.cpp
