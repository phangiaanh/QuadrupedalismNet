import torch
import numpy as np
from models.poseverts import batch_rodrigues


def mask_loss(mask_pred, mask_gt):
    """
    Input:
      mask_pred: Batch x 3 x H x W
      mask_gt: Batch x H x W
    """

    return torch.nn.L1Loss()(mask_pred, mask_gt)

def delta_v_loss(delta_v, delta_v_gt):
    """
    Input:
      mask_pred: unknown
      mask_gt: unknown
    """

    return torch.nn.L1Loss()(delta_v, delta_v_gt)

def model_trans_loss(trans_pred, trans_gt):
    """
    trans_pred: B x 3
    trans_gt: B x 3
    """
    criterion = torch.nn.MSELoss()
    return criterion(trans_pred, trans_gt)

def model_pose_loss(pose_pred, pose_gt, cfg):
    """
    pose_pred: B x 115
    pose_gt: B x 115
    """
    
    # Convert each angle in 
    R = torch.reshape( batch_rodrigues(torch.reshape(pose_pred, [-1, 3]), opts=cfg), [-1, 35, 3, 3])
    # Loss is acos((tr(R'R)-1)/2)
    Rgt = torch.reshape( batch_rodrigues(torch.reshape(pose_gt, [-1, 3]), opts=cfg), [-1, 35, 3, 3])
    RT = R.permute(0,1,3,2)
    A = torch.matmul(RT.view(-1,3,3),Rgt.view(-1,3,3))
    # torch.trace works only for 2D tensors
    n = A.shape[0]
    po_loss =  0    
    eps = 1e-7
    for i in range(A.shape[0]):
        T = (torch.trace(A[i,:,:])-1)/2.
        po_loss += torch.acos(torch.clamp(T, -1 + eps, 1-eps))
    po_loss = po_loss/(n*35)
    return po_loss

def kp_12_loss(kp_pred, kp_gt):
    """
    L2 loss between visible keypoints.

    \Sum_i [0.5 * vis[i] * (kp_gt[i] - kp_pred[i])^2] / (|vis|)
    """
    criterion = torch.nn.MSELoss()

    vis = (kp_gt[:, :, 2, None] > 0).float()

    # This always has to be (output, target), not (target, output)
    return criterion(vis * kp_pred, vis * kp_gt[:, :, :2])

