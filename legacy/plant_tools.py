

def calculate_shift_with_marker(im1, im2):
    """
    Calculate shift from a marker added to the images
    """
#     with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(np.stack([im1, im2], axis=0)
    viewer.add_points(name='point1', face_color='red', opacity=0.9, symbol='square', size=1, n_dimensional=False)

    p1, p2 = viewer.layers['point1'].__dict__['_data'][:,1:]
    p1y, p1x = p1[0], p1[1]
    p2y, p2x = p2[0], p2[1]

    manual_shift = [int(round(i)) for i in [p2y - p1y, p2x - p1x]]
    return manual_shift

def calculate_shift(im1, im2):
    """
    Calculate shift from phase cross correlation providing two images
    """
    shift, error, diffphase = registration.phase_cross_correlation(im1, im2)
#     print(shift, error, diffphase)
    return [int(i) for i in shift]

def shiftIM(im, shift):
    """
    Return the image input shifted in Y and X with the shift provided
    """
    Xshifted = np.roll(im, shift=shift[1], axis=1)
    Yshifted = np.roll(Xshifted, shift=shift[0], axis=0)
    return Yshifted

def registration_checkpoint(im1, im2, shift, savein=None):
    """
    Plot the result overlap of the 2 images, raw and shifted 
    """
    fig, ax = plt.subplots(ncols=2, figsize=(20,10))
    ax[0].set_title("Raw")
    ax[0].imshow(im1, alpha=0.5, cmap='Reds')
    ax[0].imshow(im2, alpha=0.5, cmap='Greens')
    
    im2_shifted = shiftIM(im2, shift)
    ax[1].set_title("Shifted")
    ax[1].imshow(im1, alpha=0.5, cmap='Reds')
    ax[1].imshow(im2_shifted, alpha=0.5, cmap='Greens')
    
    for i in ax:
        i.axis('off')
    
    if savein != None:
        plt.savefig(savein, dpi=300, bbox_inches = 'tight')

