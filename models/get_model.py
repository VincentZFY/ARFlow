from .pwclite import PWCLite
from .raft.raft import RAFT

def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWCLite(cfg)
    elif cfg.type == 'raft':
        model = RAFT(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
