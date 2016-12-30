/* Code written by Yang Song (yangsong@cs.stanford.edu)
 * Crop the image and do data augmentation (flip the image horizontally)
*/
#include "Image.h"
#include "Conf.h"
#include "CImg.h"
#include "database.h"
void Image::crop(int id, ValueType* dataPtr, bool do_mirror){
    cimg_library::CImg<char> image(imdb::ims + Conf::crop_size * Conf::crop_size * 3 * id
                                   , Conf::crop_size, Conf::crop_size ,1,3,true);
    if(do_mirror)
        image.mirror("x");
    for(int c = 0; c < 3; c++)
        for(int i = 0; i < Conf::crop_size; i++)
            for(int j = 0; j < Conf::crop_size; j++)
                dataPtr[c * Conf::crop_size * Conf::crop_size + i * Conf::crop_size + j] =
                         image.data()[c * Conf::crop_size * Conf::crop_size + i * Conf::crop_size + j];
}
