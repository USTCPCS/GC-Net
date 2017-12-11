from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import Variable
import torch.nn as nn

class ReqLoss(nn.Module):
    def __init__(self):
        super(ReqLoss, self).__init__()

    def forward(self, input, target):
        mask = torch.gt(target,0).float()
        n=torch.sum(mask)
        input=torch.mul(input,mask)
        loss = torch.sum(torch.abs(input-target))/n
        reutrn loss


class Validation(nn.Module):
    def __init__(self):
        super(Validation, self).__init__()
    def forward(self,input, target):
        mask=torch.gt(target,0).float()
        n=torch.sum(mask)
        input=torch.mul(input,mask)
        abs_loss=torch.abs(input-target)

        error_gt_2 = torch.sum(torch.gt(abs_loss, 2).float())

        error_gt_3 = torch.sum(torch.gt(abs_loss, 3).float())

        error_gt_5 = torch.sum(torch.gt(abs_loss, 5).float())
        return error_gt_2, error_gt_3, error_gt_5

if __name__=='__main__':
    print('...loss')
    input=torch.FloatTensor([[1,3],[2,1]])
    target=torch.FloatTensor([[0,1],[8,0]])
    input, target=Variable(input),Variable(target)
    criterion=Validation()
    mask=criterion(input,target)
    print(mask)
