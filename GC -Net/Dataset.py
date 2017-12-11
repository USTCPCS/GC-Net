from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset, DataLoader
import os
from time import time
from PIL import Image
import numpy as np

from Config import config
from Readpfm import load_pfm


H = config['Height']
C_H = config['Crop_Height']
W = config['Width']
C_W = config['Crop_Width']
D = config['M_D']
CPPI = config['Crop_patch_per_images']
n_colors = 3


class Kitty2015DataSet(Dataset):
    def __init__(self, root_dir,is_validation= False, is_training= config['is_training'], is_noc= True):
        super(Kitty2015DataSet, self).__init__()
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
        image_pad = image.crop((0, 0, W, H))
        image_crop = image_pad.crop((l_w, l_h, l_w + C_W, l_h + C_H))
        # image_crop.show()
        image_arr = np.array(image_crop)
        return image_arr

    def __len__(self):
        if self.is_training:
            if not self.is_validation:
                return len(self.filelist[0:config['number_train']]) * CPPI
            else:
                return len(self.filelist[config['number_train']:]) * CPPI
        else:
            return len(self.filelist) * CPPI

    def __getitem__(self, idx):
        if not self.is_validation:
            # idx = int(idx / CPPI)
            idx = np.random.randint(0, config['number_train'])
        else:
            # idx = int(idx / CPPI) + config['number_train']
            idx = np.random.randint(config['number_train'], len(self.filelist))
            # print(idx)
        if self.is_training:
            located_h, located_w = np.random.randint(0, H - C_H), np.random.randint(0, W - C_W)

            left_image = Image.open(os.path.join(self.path_image2, self.filelist[idx].split('.')[0]+'.png'))
            left_image_pre = self.crop_images(left_image, located_h, located_w)
            left_image = self.crop_images(left_image, located_h, located_w).transpose((2,0,1))
            left_image = left_image - np.mean(left_image)
            left_image = left_image / np.std(left_image)
            # print(left_image.shape)

            right_image = Image.open(os.path.join(self.path_image3, self.filelist[idx].split('.')[0]+'.png'))
            right_image_pre = self.crop_images(right_image, located_h, located_w)
            right_image = self.crop_images(right_image, located_h, located_w).transpose((2,0,1))
            right_image = right_image - np.mean(right_image)
            right_image = right_image / np.std(right_image)
            # print(right_image.shape)

            # disp = Image.open(os.path.join(self.path_disp, self.filelist[idx]))
            # disp = self.crop_images(disp, located_h, located_w) / 256.0
            disp, _ = load_pfm(os.path.join(self.path_disp, self.filelist[idx]))
            disp = disp[located_h:located_h+C_H, located_w:located_w+C_W].astype('float64')
            # print(disp.dtype)
            # print(disp[100])
            # disp_img = Image.fromarray(disp)
            # disp_img.show()
            # print(disp.shape)

            sample = {'left_image': left_image
                      ,'right_image': right_image
                      ,'disp': disp
                      ,'pre_left': left_image_pre
                      ,'pre_right': right_image_pre
                      }

        else:
            left_image = Image.open(os.path.join(self.path_image2, self.filelist[idx]))
            left_image = np.array(left_image)
            right_image = Image.open(os.path.join(self.path_image3, self.filelist[idx]))
            right_image = np.array(right_image)
            sample = {
                'left_image':left_image
                ,'right_image':right_image
            }

        return sample

class SceneFlow(Dataset):
    def __init__(self, if_val=False):
        super(SceneFlow, self).__init__()
        self.root_dir = config['root_dir']
        self.if_val = if_val
        self.get_file_list()

    def get_file_list(self):
        self.image_dir = os.path.join(self.root_dir, 'frames_cleanpass')
        self.disp_dir = os.path.join(self.root_dir, 'disparity')

        focallength = ['15mm_focallength', '35mm_focallength']
        direction = ['scene_backwards', 'scene_forwards']
        speed = ['slow']

        self.paths = []
        self.filepathlist = []

        for fl in focallength:
            for dt in direction:
                for sp in speed:
                    self.paths.append(os.path.join(fl, dt, sp))

        for p in self.paths:
            filename_list = os.listdir(os.path.join(self.image_dir, p, 'left'))
            for filename in filename_list:
                fileindex = filename.split('.')[0]
                self.filepathlist.append(os.path.join(p, fileindex))

        self.filepathlist = np.array(self.filepathlist)
        np.random.shuffle(self.filepathlist)
        num_train = int(len(self.filepathlist) * (1-config['val_percent']))
        if self.if_val:
            self.filepathlist = self.filepathlist[num_train:]
        else:
            self.filepathlist = self.filepathlist[0:num_train]
        self.len_fl = len(self.filepathlist)


    def crop_images(self, image, l_h, l_w):
        image_crop = image.crop((l_w, l_h, l_w + C_W, l_h + C_H))
        image_arr = np.array(image_crop)
        return image_arr

    def __len__(self):
        return len(self.filepathlist) * CPPI

    def get_true_path(self, filepath):
        dirs = filepath.split('/')
        l = len(dirs)
        tmp = dirs[l-1]
        dirs[l-1] = dirs[l-2]
        dirs[l-2] = tmp
        path = ''
        for dir in dirs:
            path = os.path.join(path, dir)
        return path

    def load_image(self, filepath):
        filepath = self.get_true_path(filepath)
        filename = filepath + '.png'
        path = os.path.join(self.image_dir, filename)
        # print(path)
        image = Image.open(path)
        if n_colors == 1:
            image = image.convert('L')
            arr = np.array(image)
        else:
            arr = np.array(image).transpose((2,0,1))
        # image.show()
        return arr

    def load_disp(self, filepath):
        filepath = self.get_true_path(filepath)
        filename = filepath + '.pfm'
        path = os.path.join(self.disp_dir, filename)
        disp, _ = load_pfm(path)
        disp = disp.astype('float64')
        # disp = disp_to_color(disp).astype('uint8')
        # img = Image.fromarray(disp)
        # img.show()
        return disp

    def preprocess(self, image):
        image = image - np.mean(image)
        image = image / np.std(image)
        return image

    def __getitem__(self, item):
        sample = {}
        idx = np.random.randint(0, self.len_fl)
        filepath = self.filepathlist[idx]
        # print(filepath)
        left_image = self.load_image(os.path.join(filepath, 'left'))
        left_image_pre = left_image
        # print(left_image.shape)
        left_image = self.preprocess(left_image)
        right_image = self.load_image(os.path.join(filepath, 'right'))
        # print(right_image.shape)
        right_image = self.preprocess(right_image)
        disp_image = self.load_disp(os.path.join(filepath, 'left'))
        # print(disp_image.shape)

        H, W = disp_image.shape
        # print(H, W)
        h = np.random.randint(0, H - C_H)
        w = np.random.randint(0, W - C_W)
        # print(h, w)

        sample = {
            'left_image': left_image[:, h:h+C_H, w:w+C_W]
            ,'right_image': right_image[:, h:h+C_H, w:w+C_W]
            ,'disp': disp_image[h:h+C_H, w:w+C_W]
            ,'pre_left': left_image_pre[:, h:h+C_H, w:w+C_W].transpose((1,2,0))
        }
        return sample
class SceneFlow_F3(Dataset):
    def __init__(self, mode='train'):
        super(SceneFlow_F3, self).__init__()
        self.root_dir = config['sff3_root_dir']
        self.mode = mode
        self.get_filelist()

    def get_filelist(self):
        self.disp_root_dir = os.path.join(self.root_dir, 'disparity')
        self.image_root_dir = os.path.join(self.root_dir, 'frames_cleanpass')
        self.image_2_list = []
        self.image_3_list = []
        self.disp_list = []
        if self.mode == 'train' or self.mode == 'val':
            # print('train/val')
            path = os.path.join(self.image_root_dir, 'TRAIN')
            names = ['A', 'B', 'C']
            for name in names:
                path_name = os.path.join(path, name)
                dir_list = os.listdir(path_name)
                for dir in dir_list:
                    image_names = os.listdir(os.path.join(path_name, dir, 'left'))
                    for image_name in image_names:
                        image2_path = os.path.join('TRAIN', name, dir, 'left', image_name)
                        image3_path = os.path.join('TRAIN', name, dir, 'right', image_name)
                        disp_path = os.path.join('TRAIN', name, dir, 'left', image_name.split('.')[0]+'.pfm')
                        self.image_2_list.append(image2_path)
                        self.image_3_list.append(image3_path)
                        self.disp_list.append(disp_path)
            self.image_2_list.sort()
            self.image_3_list.sort()
            self.disp_list.sort()
            l = int(len(self.image_2_list) * (1 - config['val_percent']))
            if self.mode == 'train':
                self.image_2_list = self.image_2_list[:l]
                self.image_3_list = self.image_3_list[:l]
                self.disp_list = self.disp_list[:l]
            else:
                self.image_2_list = self.image_2_list[l:]
                self.image_3_list = self.image_3_list[l:]
                self.disp_list = self.disp_list[l:]
        else:
            # print('test')
            path = os.path.join(self.image_root_dir, 'TEST')
            names = ['A', 'B', 'C']
            for name in names:
                path_name = os.path.join(path, name)
                dir_list = os.listdir(path_name)
                for dir in dir_list:
                    image_names = os.listdir(os.path.join(path_name, dir, 'left'))
                    for image_name in image_names:
                        image2_path = os.path.join('TEST', name, dir, 'left', image_name)
                        image3_path = os.path.join('TEST', name, dir, 'right', image_name)
                        disp_path = os.path.join('TEST', name, dir, 'left', image_name.split('.')[0]+'.pfm')
                        self.image_2_list.append(image2_path)
                        self.image_3_list.append(image3_path)
                        self.disp_list.append(disp_path)
            self.image_2_list.sort()
            self.image_3_list.sort()
            self.disp_list.sort()
        self.len_fl = len(self.image_2_list)


    def crop_images(self, image, l_h, l_w):
        image_crop = image.crop((l_w, l_h, l_w + C_W, l_h + C_H))
        image_arr = np.array(image_crop)
        return image_arr

    def __len__(self):
        return len(self.image_2_list)

    def load_image(self, filepath):
        path = os.path.join(self.image_root_dir, filepath)
        # print(path)
        image = Image.open(path)
        if n_colors == 1:
            image = image.convert('L')
            arr = np.array(image)
        else:
            arr = np.array(image).transpose((2,0,1))
        # image.show()
        return arr

    def load_disp(self, filepath):
        path = os.path.join(self.disp_root_dir, filepath)
        disp, _ = load_pfm(path)
        disp = disp.astype('float64')
        # disp = disp_to_color(disp).astype('uint8')
        # img = Image.fromarray(disp)
        # img.show()
        return disp

    def preprocess(self, image):
        image = image - np.mean(image)
        image = image / np.std(image)
        return image

    def __getitem__(self, item):
        sample = {}
        s = time()
        idx = np.random.randint(0, self.len_fl)
        # print(self.image_2_list[idx])
        left_image = self.load_image(self.image_2_list[idx])
        left_image_pre = left_image
        # print(left_image.shape)
        left_image = self.preprocess(left_image)

        # print(self.image_3_list[idx])
        right_image = self.load_image(self.image_3_list[idx])
        e = time()
        # print(' load left image',e-s)
        s = time()
        # print(right_image.shape)
        right_image = self.preprocess(right_image)
        e = time()
        # print(' load right image',e-s)
        s = time()

        disp_image = self.load_disp(self.disp_list[idx])
        e = time()
        # print(' load disp image',e-s)
        # print(disp_image.shape)

        H, W = disp_image.shape
        # print(H, W)
        h = np.random.randint(0, H - C_H)
        w = np.random.randint(0, W - C_W)
        # print(h, w)
        concat_image = np.zeros((2 * n_colors, D, C_H, C_W))

        # s = time()
        left_image = left_image[:, h:h+C_H, w:w+C_W]
        # e = time()
        # print(' concat image',e-s)
        s = time()

        # l = []
        for d in range(D):
            # con = np.concatenate((left_image, right_image[:, h:h+C_H, max(w-d,0):w-d+C_W]), axis=0)
            # l.append(con)
            # s1 = time()
            concat_image[0:n_colors, d-1, :, :] = left_image
            # e1 = time()
            # print(' per concat1',e1-s1)
            # s1 = time()
            concat_image[n_colors:, d-1, :, C_W - (w-d+C_W-max(w-d,0)):] = right_image[:, h:h+C_H, max(w-d,0):w-d+C_W]
            # e1 = time()
            # print(' per concat2',e1-s1)
        # concat_image = np.stack(l, axis= 0)
        print(concat_image.shape)

        # e = time()
        # print(' concat image',e-s)
        # s = time()
        sample ={
            'concat_image': concat_image
            , 'disp': disp_image[h:h + C_H, w:w + C_W]
            , 'pre_left': left_image_pre[:, h:h + C_H, w:w + C_W].transpose((1, 2, 0))
        }
        e = time()
        # print(' total',e-s)
        return sample

if __name__ == '__main__':
    print('Dataset!')
    ds = Kitty2015DataSet(config['root_dir'])
    # ds = SceneFlow_F3()
    # sample = ds[0]
    # print(np.max(sample['disp']))
    # sample = ds[1]
    # print(np.max(sample['disp']))
    for i in range(10):
        e = time()
        sample = ds[i]
        s = time()
        # print(s - e)
        # print(np.max(sample['disp']))
    # sample = ds[0]['pre_left'].astype('uint8').transpose((1,2,0))
    sample = ds[0]['pre_left'].astype ( 'uint8' )
    # sample=np.array(sample)
    # print ( sample.dtype, sample.size, sample.shape )
    img =Image.fromarray(sample)
    img.show()
    sample = ds[0]
    dl = iter(DataLoader(ds, 1))
    sample1 = dl.next()
    print(sample1['left_image'].size())
    img=Image.fromarray(sample['pre_left'].astype('uint8'))
    img.show()
    pre_left = ds[0]['pre_left']
    print(pre_left.dtype)
    img_left = Image.fromarray(pre_left)
    img_left.show()