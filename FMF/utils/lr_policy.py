import torch


def get_lr_policy(optimizer, cfg):
    policy = cfg.SOLVER.LR_POLICY
    if policy == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.SOLVER.MAX_EPOCH, eta_min=cfg.SOLVER.COSINE_END_LR
        )
    elif policy == 'step':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS)
    elif policy == 'none':
        lr_policy = NoLrPolicy()
    else:
        raise ValueError('cfg.SOLVER.LR_POLICY must be cosine or step, but get {}'.format(policy))
    return lr_policy


class NoLrPolicy:
    def __init__(self):
        self.message = 'do nothing for lr'

    def step(self):
        pass

    def __repr__(self):
        return self.message
