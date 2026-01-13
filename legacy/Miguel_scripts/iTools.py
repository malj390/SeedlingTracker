from PIL import Image
from matplotlib import pyplot as plt
import math
import numpy as np
from skimage import color


normalize_im = lambda x: x/np.max(x)
normalize_im.__doc__ = "Normalize an image from its brigthest value"

def im_to_rgb(im, chs=['blue', 'red']):
    """
    im: It has to be an image with just 2 dimensions YX
    chs: ['red', 'yellow', 'blue', 'purple']
    """

    rgbim = [color.gray2rgb(normalize_im(i)) for i in im]

    rgbCode = {
       'red' : [1,0,0] ,
        'yellow' : [1,1,0] ,
        'blue' : [0,0,1] ,
        'purple' : [1,0,1]}
    colored_im = [rgbCode[chs[i]]*rgbim[i] for i in range(im.shape[0])]
    res = np.sum(colored_im, axis=0)
    return res


def tile_plot_imshow_png(images, titles=None):
    if titles == None:
        titles = range(1, len(images))
    n = math.ceil(np.sqrt(len(images)))
    fig, ax = plt.subplots(ncols=n, nrows=n, figsize=(20,20))
    ax = fig.get_axes()
    for ix, i in enumerate(ax):

        try:
            im = images[ix]
            ax[ix].imshow(im)
            ax[ix].set_title("{}".format(titles[ix]))
            ax[ix].axis('off')
        except:
            ax[ix].axis('off')


def plot_squared_pics(list_ims, title, list_names, savein=None):
    n = math.ceil(np.sqrt(len(list_ims)))
    fig, ax = plt.subplots(ncols=n, nrows=n, figsize=(10, 10), facecolor='black')
    ax = fig.get_axes()
    plt.suptitle("{}".format(title), color='white')
    plt.subplots_adjust(hspace=0.5)
    for ix, i in enumerate(ax):
        try:
            ax[ix].imshow(list_ims[ix])
            ax[ix].axis('off')
            ax[ix].set_title(list_names[ix], color='white', fontdict={"size":6})
        except:
            ax[ix].axis('off')
            
    if savein != None:
        plt.savefig(savein, dpi=600, bbox_inches="tight", facecolor='black')