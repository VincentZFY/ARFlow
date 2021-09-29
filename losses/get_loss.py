from .flow_loss import unFlowLoss, MultiScaleEPE,LossRAFT,unFlowLossRAFT

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'unFlowLossRAFT':
        loss = unFlowLossRAFT(cfg)
    elif cfg.type == 'MultiScaleEPE':
        loss = MultiScaleEPE(cfg)
    elif cfg.type == 'LossRAFT':
        loss = LossRAFT(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
