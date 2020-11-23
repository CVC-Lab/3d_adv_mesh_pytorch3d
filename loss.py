import torch
import torch.nn as nn
import numpy as np

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
    
    # TODO: Get the code of 2D Adv Patch TV Loss back as a different class/function

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


class TotalVariation_3d(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self, mesh, target_face_id):
        super(TotalVariation_3d, self).__init__()
        
        # TODO: deal with different input meshes. the code assume the mesh topology are the same now.
        # TODO: need adv_patch to initialize everything, think a better way to record topology.

        # Step 0: get all info of mesh[0]
        FtoE_id = mesh[0].faces_packed_to_edges_packed().cpu()
        EtoV_id = mesh[0].edges_packed().cpu()
        V = mesh[0].verts_packed()
        num_of_edges = EtoV_id.shape[0]
        num_of_target_faces = len(target_face_id)
        
        # Step 1: Construct (E_n, 2) tensor as opposite face indexing
        EtoF_idx1 = -1 * torch.ones((num_of_edges),dtype=torch.long)
        EtoF_idx2 = -1 * torch.ones((num_of_edges),dtype=torch.long)

        for i in range(num_of_target_faces):
            for each in FtoE_id[target_face_id[i]]:
                if EtoF_idx1[each]==-1:
                    EtoF_idx1[each] = i
                else:
                    EtoF_idx2[each] = i
        # remove all edges that does not belong to 
        valid_id = ~((EtoF_idx1 == -1) | (EtoF_idx2 == -1))

        EtoF_idx = torch.stack((EtoF_idx1[valid_id],EtoF_idx2[valid_id]), dim=1)


        # Do we need .cuda() here?
        self.face_to_edges_idx = EtoF_idx.cuda()

        # Step 2: Compute edge length
        valid_edge = EtoV_id[valid_id,:]
        self.edge_len = torch.norm(V[valid_edge[:,0],:]-V[valid_edge[:,1],:], dim=1).cuda()

        # Step 3: Now we have real id of neighboring faces and weights (edge length)
        # we can compute TV loss whenever a patch comes in.
    
    def forward(self, adv_patch):
        
        # mesh_num = len(mesh)

        # adv_patch :  assume piece-wise constant!
        f1 = adv_patch[self.face_to_edges_idx[:,0],:,:,:]
        f2 = adv_patch[self.face_to_edges_idx[:,1],:,:,:]
        tv = torch.sum(self.edge_len[:,None,None,None] * torch.abs(f1-f2))
        return tv / adv_patch.shape[0]

