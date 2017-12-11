from __future__ import print_function
from __future__ import absolute_import

config={
    'data_dir': '/home/youmi/projects/GC -Net',
    'root_dir': '/home/youmi/projects/GC -Net/KITTY-SF',
    'model_par_dir' : 'New_Parameters_SF',
    'model_name_newest' : 0,

    'batch_size': 1,
    'learning_rate': 0.001,
    'epoches': 10,
    'batches_per_epoch': 7500,
    'number_validation': 50,

    'Height': 540,
    'Width': 960,
    'M_D': 192,
    'Feature_size': 32,
    'val_percent': 0.2,

    'if_GPU': True,

    'Crop_Height': 256,
    'Crop_Width': 512,
    'Crop_patch_per_images': 4,
    'is_training': True,
    'number_train': 1
}