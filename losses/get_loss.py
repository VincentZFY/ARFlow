from .flow_loss import unFlowLoss, MultiScaleEPE

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'MultiScaleEPE':
        loss = MultiScaleEPE(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
