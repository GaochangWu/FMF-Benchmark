from .CurrentDataset import CurrentDataset
from .VideoDataset import VideoDataset
from .ViCuDataset import ViCuDataset

from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
def fmf_current(cfg, split):
    dataset = CurrentDataset(
        mode=split, prefix=cfg.DATA.PATH_TO_DATA_DIR, current_seconds=cfg.DATA.CURRENT_SECONDS,
    )
    return dataset


@DATASET_REGISTRY.register()
def fmf_video(cfg, split):
    dataset = VideoDataset(
        mode=split, task=cfg.TASK, prefix=cfg.DATA.PATH_TO_DATA_DIR,
        num_frames=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE,
    )
    return dataset


@DATASET_REGISTRY.register()
def fmf_vicu(cfg, split):
    dataset = ViCuDataset(
        mode=split, prefix=cfg.DATA.PATH_TO_DATA_DIR, task=cfg.TASK,
        num_frames=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE,
        current_seconds=cfg.DATA.CURRENT_SECONDS,
    )
    return dataset
