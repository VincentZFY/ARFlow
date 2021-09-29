import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
#from corr import CorrBlock, AlternateCorrBlock
from .corr import CorrBlock
from .utils import bilinear_sampler, coords_grid, upflow8

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.upsample = args.upsample

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # if 'alternate_corr' not in self.args:
        #     self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward_once(self, img1, fmap1, fmap2, iters=12, flow_init=None):
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(img1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)

        # if dropout is being used reset mask
        self.update_block.reset_mask(net, inp)
        coords0, coords1 = self.initialize_flow(img1)

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if self.upsample:
                flow_up = upflow8(coords1 - coords0)
                flow_predictions.append(flow_up)

            else:
                flow_predictions.append(coords1 - coords0)
        return flow_predictions[::-1]

    def forward(self, imgs, iters=12, flow_init=None, with_bk=False):
        """ Estimate optical flow between pair of frames """
        image1, image2 = imgs.chunk(2, dim=1)

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        res = {'flows_fw': self.forward_once(image1, fmap1, fmap2, iters, flow_init)}
        if with_bk:
            res['flows_bw'] = self.forward_once(image2, fmap2, fmap1, iters, flow_init)
        return res