import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def cross_entropy2d(predict, target, weight=None, size_average=True):
    # predict: (n, c, h, w), target: (n, h, w)
    n, c, h, w = predict.size()
    log_p = F.log_softmax(predict, dim=1)
    log_p = log_p.permute(0, 2, 3, 1).reshape(-1, c)
    target = target.view(-1)
    reduction = 'mean' if size_average else 'sum'

    # 计算损失
    loss = F.nll_loss(log_p, target, weight=weight, reduction=reduction)
    return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.eps = 1e-8
    def forward(self, predict, target):
        if self.weight!=None:
            weights = self.weight.unsqueeze(0).unsqueeze(1).repeat(predict.shape[0], predict.shape[2], 1)
        target_onehot = F.one_hot(target.long(), predict.shape[1]) 
        if self.weight!=None:
            weights = torch.sum(target_onehot * weights, -1)
        input_soft = F.softmax(predict, dim=1)
        probs = torch.sum(input_soft.transpose(2, 1) * target_onehot, -1).clamp(min=0.001, max=0.999)#此处一定要限制范围，否则会出现loss为Nan的现象。
        focal_weight = (1 + self.eps - probs) ** self.gamma
        if self.weight!=None:
            return torch.sum(-torch.log(probs) * weights * focal_weight) / torch.sum(weights)
        else:
            return torch.mean(-torch.log(probs) * focal_weight)
