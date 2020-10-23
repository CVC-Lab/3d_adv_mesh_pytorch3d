import torch
import torch.nn as nn

from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from darknet import Darknet

def dis_loss(output, num_classes, anchors, num_anchors, target_id=0, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)

    all_target_acc = []
    det_confs = torch.sigmoid(output[4])
    #TODO: Double chekc if this is correct
    cls_confs = F.softmax(Variable(output[5:5 + num_classes].transpose(0, 1)), -1)
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    cls_max_ids = torch.eq(cls_max_ids, target_id).float()

    det_human_conf = torch.where(cls_max_ids == 0., cls_max_ids, det_confs)
    det_human_conf = det_human_conf.contiguous().view(batch, -1)
    target_conf, target_conf_id = torch.max(det_human_conf, 1)

    return torch.mean(target_conf)

def calc_acc(output, num_classes, num_anchors, target_id):
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)

    all_target_acc = []
    det_confs = torch.sigmoid(output[4])
    
    #TODO: Double check if this is correct
    cls_confs = F.softmax(Variable(output[5:5 + num_classes].transpose(0, 1)), -1)
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    cls_max_ids = torch.eq(cls_max_ids, target_id).float()

    det_human_conf = torch.where(cls_max_ids == 0., cls_max_ids, det_confs)
    det_human_conf = det_human_conf.contiguous().view(batch, -1)

    target_conf, target_conf_id = torch.max(det_human_conf, 1)
    target_conf = target_conf.detach().cpu().data
    count = torch.sum(target_conf < 0.6).float().data

    return count

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """
    
    def __init__(self):
        super(TotalVariation, self).__init__()
    
    def forward(self, adv_patch):
        # adv_patch : (H, W, C)
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, 1:, :, :] - adv_patch[:, :-1, :, :] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(torch.sum(tvcomp1, 0), 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, :, 1:, :] - adv_patch[:, :, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(torch.sum(tvcomp2, 0), 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)
