import pandas as pd

def writting_excel(DF, pathname, sheet_name=None):
    writer = pd.ExcelWriter(pathname, engine='xlsxwriter')
    if type(DF) != type([]):
        if sheet_name == None:
            DF.to_excel(writer, index=None)
        else:
            DF.to_excel(writer, sheet_name=sheet_name)
        writer.save()
    else:
        for ix, each in enumerate(DF):
            if sheet_name == None:
                DF[ix].to_excel(writer, header=None, index=None)
            else:
                DF[ix].to_excel(writer, sheet_name=sheet_name[ix])
        writer.save()
        
        
from PIL import Image
from PIL.TiffTags import TAGS

def extract_metadata_TiFF(path, short=True):
    """
    Extract metadata from the image path
    spacing = Z um
    XResolution = X um
    YResolution = Y um
    unit = um
    """
    with Image.open(str(path)) as img:
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
        
    if short == True:
        dict_description = dict([i.split("=") for i in meta_dict['ImageDescription'][0].split("\n") if "=" in i])
        dict_description['spacing'] = float(dict_description['spacing'])
        YResolution, XResolution = [meta_dict[i][0][1]/meta_dict[i][0][0] for i in ['YResolution', 'XResolution']]
        dict_description['YResolution'], dict_description['XResolution'] =  YResolution, XResolution
        return dict_description
    else:
        return meta_dict

import napari
    
def napariView(image, name='', label_im=None):
    if type(name) != type([]) and type(image) == type([]): name = ["Image_{}".format(i) for i in range(len(image))]
    
    with napari.gui_qt():
        viewer = napari.Viewer()
        if type(image) == type([]):
            for ix, im in enumerate(image):
                if type(label_im) != type(None):
                    viewer.add_image(image[ix], name="{}".format(name[ix]))
                    viewer.add_labels(label_im[ix], name="Label_{}".format(name[ix]))
                else:
                    viewer.add_image(image[ix], name="{}".format(name[ix]))
        else:
            if type(label_im) != type(None):
                viewer.add_image(image, name="{}".format(name))
                viewer.add_labels(label_im, name="Label_{}".format(name))
            else:
                viewer.add_image(image, name="{}".format(name))
                
        
        
        
        
#import h5py as h
#
#def get_h5_dataset(fp, dset_name):
#    with h.File(fp, 'r') as f:
#        assert dset_name in f.keys(), f"dataset {dset_name} does not exist. Datasets are: {[k for k in f.keys()]}"
#        data = f.get(dset_name)[:]
#    return data        
#        

sublist = lambda lst, chunk_size: [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
sublist.__doc__ = "Provide lst and chunk_size, it returns the list splitted in chunks of the specified size"
        
def sublist_unequally(lst, sizes):
    """
    Provide list LST and sizes of the chunks to get. 
    If chunks are not done for the amount available, 
    those will be excluded and it will raise and warning
    """
    it = iter(lst)
    result =  [[next(it) for _ in range(size)] for size in sizes]
    items_in_result = len([i for u in result for i in u])
    left = len(lst) - items_in_result
    info = "{} from {} where sublisted. {} left excluded".format(items_in_result, len(lst), left)
    print(info)
    return result        
        
        
def print_pretty_list(L):
    for ix, i in enumerate(L):
        print(ix, i)        
        
        
        
