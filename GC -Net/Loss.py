from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import Variable
import torch.nn as nn


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, input, target):
        mask = torch.gt(target, 0).float()
        n = torch.sum(mask)
        input = torch.mul(input, mask)
        loss = torch.sum(torch.abs(input - target)) / n
        return loss


class Validation(nn.Module):
    def __init__(self):
        super(Validation, self).__init__()

    def forward(self, input, target):
        mask = torch.gt(target, 0).float()
        n = torch.sum(mask)
        input = torch.mul(input, mask)
        abs_loss = torch.abs(input - target)
        # error_lt_1 = torch.sum(torch.le(abs_loss, 1).float() - torch.eq(abs_loss, 0).float()) / n
        error_gt_2 = torch.sum(torch.gt(abs_loss, 2).float()) / n
        # error_lt_3 = torch.sum(torch.le(abs_loss, 3).float() - torch.eq(abs_loss, 0).float()) / n
        error_gt_3 = torch.sum(torch.gt(abs_loss, 3).float()) / n
        # error_lt_5 = torch.sum(torch.le(abs_loss, 5).float() - torch.eq(abs_loss, 0).float()) / n
        error_gt_5 = torch.sum(torch.gt(abs_loss, 5).float()) / n
        return error_gt_2, error_gt_3, error_gt_5


if __name__ == '__main__':
    print('...Loss')
    input = torch.FloatTensor([[1, 3], [2, 1]])
    target = torch.FloatTensor([[0, 1], [8, 0]])
    input, target = Variable(input), Variable(target)
    criterion = Validation()
    mask = criterion(input, target)
    print(mask)
