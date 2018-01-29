import numpy as np
from scipy.misc import imread, imresize, imsave


def get_image(path, shape, resize_len=None, rand_crop=False):
    '''
    shape: [height, width], correct shape to feed in the network
    return: processed image as shape of SHAPE
    '''
    img=imread(path)
    height, width, _ = img.shape

    if rand_crop:
        if resize_len==None:
            resize_len=shape[0]*2 #
        if height < width:
            new_height = resize_len
            new_width  = int(width * new_height / height)
        else:
            new_width  = resize_len
            new_height = int(height * new_width / width)
        # random crop
        img= imresize(img, [new_height, new_width], interp='nearest')
        start_h = np.random.choice(new_height - shape[0] + 1)
        start_w = np.random.choice(new_width - shape[1] + 1)
        img = img[start_h:(start_h + shape[0]), start_w:(start_w + shape[1]), :]
    else:
        img=imresize(img, size=shape)
    return img

def get_train_images(paths, shape, resize_len, rand_crop=False):
    images=[]
    for path in paths:
        images.append(get_image(path, shape, resize_len, rand_crop))
    images = np.stack(images, axis=0)
    return images