#ifndef IMAGE_CROP_HEADER
#define IMAGE_CROP_HEADER
#include "Conf.h"
using std::pair;
using std::make_pair;
using std::string;
using std::vector;
class Image{
public:
   static void crop(int id, ValueType *dataPtr, bool do_mirror = true);
};

#endif // IMAGE_CROP
