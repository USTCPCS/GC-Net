from __future__ import print_function
from __future__ import absolute_import

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torch.autograd import Variable
import numpy as np
from PIL import Image
import warnings
import time

from my_network import GCNet
from my_Dataset import Kitty2015DataSet
from my_loss import ReqLoss, Validation
from my_config import config

class Model(object):
    def __init__(self):
        self.data_patch=config['data_dir']
        self.model_path=os.path.join(config['model_par_dir'],str(config['model_name_newest'])+'.pkl')
        self.build_net()
        self.load_par()
        self.model_name_newest=config['model_name_newest']

    def build_net(self):
        self.net=GCNet()
        if config['if_GPU']:
            self.net=self.net.cuda()
        self.dataset={}
        self.dataloader={}
        self.dataset['train']=Kitty2015DataSet()
        self.dataset['val']=Kitty2015DataSet(if_val=True)
        self.dataloader['train']=iter(
            DataLoader(dataset=self.dataset['train'],batch_size=config['batch_size'],shuffle=True)
        )
        self.dataloader['val']=iter(
            DataLoader(dataset=self.dataset['val'],batch_size=config['batch_size'],shuffle=True)
        )
        self.criterion=RegLoss()
        self.val=Validation()
        self.opti=RMSprop(self.net.parameters(),lr=config['learning_rate'])

    def load_par(self):
        if os.path.exists(self.model_path):
            self.net.load_state_dict(torch.load(self.model_path))

    def save_par(self):
        model_name=str(self.model_name_newest)+'.pkl'
        model_path=os.path.join(config['model_par_dir'],model_name)
        torch.save(self.net.state_dict(),model_path)
        config['model_name_neweset']=self.model_name_newest
        self.model_name_newest+=1
        self.model_name_newest%=5

    def train(self):
        epoches=config['epoches']
        start_time=time.time()
        for epoch in range(epoches):
            total_loss=0
            for batch_index in range(config['bathces_per_epoch']):
                st=time.time()
                sample=iter(self.dataloader['train']).next()
                loss=self.train_batch(sample,epoch,batch_index)
                et=time.time()
                print('cost {0} seconds'.format(int(et-st+0.5)))
                total_loss+=loss

                if (batch_index+1)%config['batch_per_validation']==0:
                    end_time=time.time()
                    print('....time:{0}'.format(end_time-start_time))
                    start_time=end_time
                    print('...epoch : {0:2d}'.format(epoch))
                    print('...validation')

                    v0_t,v1_t,v2_t=0,0,0
                    num=config['number_validation']
                    for i in range(num):
                        sample=iter(self.dataloader['val']).next()
                        v0,v1,v2=self.validation(sample)
                        v0_t+=v0
                        v1_t+=v1
                        v2_t+=v2
                    print('...>2 px : {0}%   >3px : {1}%   >5px : {2}%'.format(v0_t/num,v1_t/num,v2_t/num))
                    print('...save parameters')

                print('... average loss : {0}'.format(total_loss/config['batches_per_epoch']))

    def train_batch(self, batch_sample, epoch, batch_index):
        left_image=batch_sample['left_image'].float()
        right_image=batch_sampel['right_image'].float()
        disp=batch_sample['disp'].float()
        if config['if_GPU']:
            left_image=left_image.cuda()
            right_image=right_image.cuda()
            disp=disp.cuda()
        left_image,right_image,disp=Variable(left_image),Variable(right_image), Variable(disp)
        disp_prediction=self.net((left_image,right_image))
        if(config['if_GPU']):
            disp_prediction=disp_prediction.cuda()
        loss=self.criterion(disp_prediction,disp)
        if config['if_GPU']:
            loss=loss.cuda()
        print('...epoch : {1} batch_index : {2} loss : {0}'.format(loss.data[0],epoch,batch_index))
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()
        return loss.data[0]

    def validation(self, batch_sample):
        left_image=batch_sample['left_image'].float()
        right_image=batch_sample['right_image'].float()
        disp=batch_sample['disp'].float()
        if config['if_GPU']:
            left_image=left_image.cuda()
            right_image=right_image.cuda()
            disp=disp.cuda()
        left_image,right_image,disp=Variable(left_image),Variable(right_image),Variable(disp)
        disp_prediction=self.net((left_image,right_image))
        if(config['if_GPU']):
            disp_prediction=disp_prediction.cuda()
        val=self.val(disp_prediction,disp)
        if config['if_GPU']:
            val=[val[0].cuda(), val[1].cuda(), val[2].cuda()]

        return val[0].data[0]*100, val[1].data[0]*100, val[2].data[0]*100

    def predict_batch(self, batch_sample):
        left_image=batch_sample['left_image'].float()
        right_image=batch_sample['right_image'].float()
        disp=batch_sample['disp'].float()
        if config['if_GPU']:
            left_image=left_image.cuda()
            right_image=right_image.cuda()
            disp=disp.cuda()

        left_image,right_image,disp=Variable(left_image),Variable(right_image),Variable(disp)
        disp_prediction=self.net(left_image,right_image)
        loss=self.criterion(disp_prediction,disp)
        if config['if_GPU']:
            loss=loss.cuda()
        print('loss : {0}'.format(loss.data[0]))
        return disp,disp_prediction

    def predict(self):
        for i in range(10):
            print(i)
            dir=os.path.join('./Data',str(i))
            if not os.path.exists(dir):
                os.makedirs(dir)
            batch_sample=self.dataloader['val'].next()
            left_image_pre=batch_sample['pre_left']
            image_pre=left_image_pre[0].numpy()
            disp,disp_prediction=self.predict_batch(batch_sample)
            disp_prediction=disp_prediction.cpu().data.numpy()*256
            disp_gt=disp.cpu().data.numpy()[0]*256
            print(disp_prediction.shape)
            np.save(os.path.join(dir,'image_pre.npy'), image_pre)
            np.save(os.path.join(dir,'disp_gt.npy'),disp_gt)
            np.save(os.path.join(dir,'disp_est.npy'),disp_prediction)


if __name__=='__main__':
    print('...gc-net main!')
    model=Model()
    model.train()

