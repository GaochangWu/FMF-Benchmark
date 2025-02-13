from .build import MODEL_REGISTRY

from .VitForVideo import ViTForVideo
from .VitForCurrent import ViTForCurrent
from .VitForViCu import ViTForViCu


@MODEL_REGISTRY.register()
def vit_for_video(cfg):
    model = ViTForVideo(
        image_size=cfg.DATA.IMAGE_SIZE, frames=cfg.DATA.NUM_FRAMES, num_classes=cfg.MODEL.NUM_CLASSES,
        image_patch_size=cfg.MODEL.IMAGE_PATCH_SIZE, frame_patch_size=cfg.MODEL.FRAME_PATCH_SIZE,
        dim=96, depth=6, heads=3, mlp_dim=4 * 96, pool='cls', dim_head=32, channels=3, dropout=0., emb_dropout=0.,
        PEType=cfg.MODEL.PETYPE, task=cfg.TASK, additional=cfg.MODEL.ADDITIONAL_TOKENS,
    )
    return model


@MODEL_REGISTRY.register()
def vit_for_current(cfg):
    model = ViTForCurrent(
        num_patches=2 * cfg.DATA.CURRENT_SECONDS, patch_dim=cfg.DATA.CURRENT_PATCH_DIMS,
        num_classes=cfg.MODEL.NUM_CLASSES,
        dim=96, depth=6, heads=3, mlp_dim=4 * 96, pool='cls', dim_head=32, dropout=0., emb_dropout=0.
    )
    return model


@MODEL_REGISTRY.register()
def vit_for_vicu(cfg):
    model = ViTForViCu(
        image_size=cfg.DATA.IMAGE_SIZE, frames=cfg.DATA.NUM_FRAMES, num_classes=cfg.MODEL.NUM_CLASSES,
        image_patch_size=cfg.MODEL.IMAGE_PATCH_SIZE, frame_patch_size=cfg.MODEL.FRAME_PATCH_SIZE,
        current_patches=2 * cfg.DATA.CURRENT_SECONDS, current_patch_dim=cfg.DATA.CURRENT_PATCH_DIMS,
        task=cfg.TASK, additional=cfg.MODEL.ADDITIONAL_TOKENS,
        dim=96, depth=6, heads=3, mlp_dim=4 * 96, pool='cls', dim_head=32, channels=3, dropout=0., emb_dropout=0.,
    )
    return model
