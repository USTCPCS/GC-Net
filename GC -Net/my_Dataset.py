from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset, DataLoader
import os
from time import time
from PIL import Image
import numpy as np

from my_config import config
from Readpfm import load_pfm

H = config['Height']
C_H = config['Crop_Height']
W = config['Width']
C_W = config['Crop_Width']
D = config['M_D']
CPPI = config['Crop_patch_per_images']
n_colors = 3

class Kitty2015Dataset(Dataset):
    def __init__(self, root_dir , is_validation=False , is_training=config['is_training'], is_noc=True):
        super(Kitty2015Dataset,self).__init__()
        self.root_dir = root_dir
        self.is_validation = is_validation
        self.is_training = is_training
        self.is_noc = is_noc
        self.get_filelist()

    def get_filelist(self):
        if self.is_training:
            self.path_tr = os.path.join(self.root_dir, 'training')
            self.path_image2 = os.path.join(self.path_tr, 'image_2')
            self.path_image3 = os.path.join(self.path_tr, 'image_3')
            if self.is_noc:
                self.path_disp = os.path.join(self.path_tr, 'disp_noc_0')
            else:
                self.path_disp = os.path.join(self.path_tr, 'disp_occ_0')
            self.filelist = os.listdir(self.path_disp)
        else:
            self.path_te = os.path.join(self.root_dir, 'test')
            self.path_image2 = os.path.join(self.path_te, 'image_2')
            self.path_image3 = os.path.join(self.path_te, 'image_3')
            self.filelist = os.listdir(self.path_image2)

    def crop_images(self, image, l_h, l_w):
        image_pad = image.crop((0,0,W,H))
        image_crop = image_pad.crop((l_w, l_h, l_w+C_W, l_h+C_H))

        image_arr=np.array(image_crop)
        return image_arr

    def __len__(self):
        if self.is_training:
            if not self.is_validation:
                return len(self.filelist[0:config['number_train']])*CPPI
            else:
                return len(self.filelist[config['number_train']:])*CPPI
        else:
            return len(self.filelist)*CPPI

    def __getitem__(self,idx):
        if not self.is_validation:
            idx = np.random.randint(0, config['number_train'])
        else:
            idx = np.random.randint(config['number_train'],len(self.filelist))

        if self.is_training:
            located_h, located_w =np.random.randint(0, H-C_H), np.random.randint(0, W-C_W)

            left_image = Image.open(os.path.join(self.path_image2, self.filelist[idx].split('.')[0]+ '.png'))
            left_image_pre =self.crop_images(left_image, located_h, located_w)
            left_iamge = self.crop_images(left_image, located_h, located_w).transpose((2,0,1))
            # left_image_pre.show()
            left_image = left_image - np.mean(left_image)
            left_iamge = left_iamge / np.std(left_image)

            right_image = Image.open(os.path.join(self.path_image3, self.filelist[idx].split('.')[0]+'.png'))
            right_image_pre = self.crop_images(right_image,located_h, located_w)
            right_image = self.crop_images(right_image,located_h,located_w).transpose((2,0,1))
            # right_image.show()
            right_image = right_image -np.mean(right_image)
            right_image = right_image / np.std(right_image)

            disp , _ =load_pfm(os.path.join(self.path_disp, self.filelist[idx]))
            disp = disp[located_h:located_h+C_H, located_w:located_w+C_W]

            sample={'left_image': left_iamge,
                    'right_image': right_image,
                    'disp': disp,
                    'pre_left': left_image_pre,
                    'pre_right': right_image_pre}
        else:
            left_image = Image.open(os.path.join(self.path_image2, self.filelist[idx]))
            left_image = np.array(left_image)
            right_image = Image.open(os.path.join(self.path_image3, self.filelist[idx]))
            right_image = np.array(right_iamge)
            sample={'left_image': left_iamge,
                    'right_image': right_image}

        return sample

if __name__ =='__main__':
    print('Dataset')
    ds=Kitty2015Dataset(config['root_dir'])

    for i in range(100):
        e=time()
        sample =ds[i]
        s=time()
        # print(s-e)
    sample=ds[0]['pre_left'].astype('uint8')
    img=Image.fromarray(sample)
    img.show()
