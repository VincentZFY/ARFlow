import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                           (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                           (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)

            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        total_loss = warp_loss + smooth_loss

        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()

class MultiScaleEPE(nn.modules.Module):
    def __init__(self, cfg):
        super(MultiScaleEPE, self).__init__()
        self.cfg = cfg
        self.w_level = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output, target):
        """

        :param output: n * [B x 2 x h x w]
        :param target: B x 2 x H x W
        :return:
        """

        pyramid_flows = output
        flow_gt = target * self.cfg.div

        pyramid_flow_losses = []
        pyramid_epe = []
        for i, flow_pred in enumerate(pyramid_flows):
            b, _, h, w = flow_pred.size()
            _, _, H, W = flow_gt.size()
            flow_pred[:, 0] = flow_pred[:, 0] / w * W * self.cfg.div
            flow_pred[:, 1] = flow_pred[:, 1] / h * H * self.cfg.div
            flow_gt_scaled = F.interpolate(flow_gt, (h, w), mode='area')
            loss_flow = elementwise_loss(flow_pred, flow_gt_scaled, p=self.cfg.p,
                                         eps=self.cfg.eps, q=self.cfg.q).sum() / b
            epe = elementwise_loss(flow_pred, flow_gt_scaled, p=2, eps=1e-7, q=1).mean()
            pyramid_flow_losses.append(loss_flow)
            pyramid_epe.append(epe)

        total_loss = sum([l * w for l, w in zip(pyramid_flow_losses, self.w_level)])
        return total_loss, pyramid_epe[:4]

def elementwise_loss(pred, gt, p=2, eps=0.01, q=0.4, mask=None):
    diff = pred - gt
    loss_map = torch.pow(torch.norm(diff, p=p, dim=1, keepdim=True) + eps, q)
    if mask is not None:
        loss_map *= mask
    return loss_map

class LossRAFT(nn.modules.Module):
    def __init__(self, cfg):
        super(LossRAFT, self).__init__()
        self.cfg = cfg

    def forward(self, output, target):
        flow_list = output
        flow_gt = target * self.cfg.div
        b, _, h, w = flow_list[0].size()
        _, _, H, W = flow_gt.size()
        flow_gt_scaled = F.interpolate(flow_gt, (h, w), mode='area')

        # exlude invalid pixels and extremely large diplacements
        valid = (flow_gt_scaled.abs().sum(dim=1) < 1000).float().unsqueeze(1)

        pyramid_flow_losses = []
        pyramid_epe = []
        for i, flow_pred in enumerate(flow_list):
            flow_pred[:, 0] = flow_pred[:, 0] / w * W * self.cfg.div
            flow_pred[:, 1] = flow_pred[:, 1] / h * H * self.cfg.div
            loss_flow = elementwise_loss(flow_pred, flow_gt_scaled, p=self.cfg.p,
                                         eps=self.cfg.eps, q=self.cfg.q, mask=valid).mean()
            epe = elementwise_loss(flow_pred, flow_gt_scaled, p=2, eps=1e-7, q=1).mean()
            pyramid_flow_losses.append(loss_flow)
            pyramid_epe.append(epe)

        total_loss = sum(
            [l * self.cfg.w_gama ** i for i, l in enumerate(pyramid_flow_losses)])
        return total_loss, pyramid_epe[:4]

class unFlowLossRAFT(unFlowLoss):
    def __init__(self, cfg):
        super(unFlowLossRAFT, self).__init__(cfg)
        self.w_gama = cfg.w_gama

    def forward(self, output, target):
        pyramid_flows = output
        b, _, h, w = pyramid_flows[0].size()


        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        # resize images to match the size of layer
        im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
        im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

        total_loss_warp = total_loss_smooth = 0
        for i, flow in enumerate(pyramid_flows):
            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)


            if self.cfg.occ_from_back:
                occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
            else:
                occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:,:2])
            
            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            s = min(h, w) * 8
            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)
            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            weight = self.w_gama ** i  # (len(flows) - 1 - i)
            total_loss_warp += weight * loss_warp
            total_loss_smooth += weight * self.cfg.w_smooth * loss_smooth
        total_loss = total_loss_warp + total_loss_smooth
        return total_loss, total_loss_warp, total_loss_smooth, pyramid_flows[0].abs().mean()
