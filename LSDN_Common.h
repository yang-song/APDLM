//Author: Alexander G. Schwing (http://alexander-schwing.de)
#ifndef __LSDN_COMMON_H__
#define __LSDN_COMMON_H__

enum STATE : unsigned char { TRAIN, TRAIN_FUNEVAL, TEST, VALIDATE };
enum NODETYPE : unsigned char { NODE_UNDEF, NODE_PARAM, NODE_DATA, NODE_FUNC };

template<bool U> struct i2t {};

#endif
