import numpy as np
import random

import torch
import torch.nn.functional as F


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip, masks=None):
        """
        :param clip: np.array, shape of [..., H, W]
        :param masks: np.array, shape of [..., H, W]
        :return:
        """
        if random.uniform(0, 1) < self.p:
            clip = np.ascontiguousarray(clip[..., :, ::-1])
            if masks is not None and type(masks) is np.ndarray:
                masks = np.ascontiguousarray(masks[..., :, ::-1])

        if masks is None:
            return clip
        else:
            return clip, masks


class RandomSizedCrop:
    def __init__(self, scale=(0.6, 1.0), out_h=6, out_w=6, p=1):
        self.scale = scale
        self.out_h = out_h
        self.out_w = out_w
        self.p = p

    def __call__(self, clip, masks=None):
        """
        :param clip: shape - C T H W
        :param masks: shape - H W
        :return:
        """
        if random.uniform(0, 1) < self.p:
            clip, masks = self.crop(clip, masks)
            clip, masks = self.resize(clip, masks)
        if masks is not None:
            return clip, masks
        else:
            return clip

    def crop(self, clip, masks):
        scale_h = random.uniform(self.scale[0], self.scale[1])
        scale_w = random.uniform(self.scale[0], self.scale[1])

        H, W = clip.shape[-2:]
        crop_h, crop_w = int(H * scale_h), int(W * scale_w)

        start_h = random.randint(0, H - crop_h)
        start_w = random.randint(0, W - crop_w)

        clip = clip[..., start_h:start_h + crop_h, start_w:start_w + crop_w]
        if masks is not None:
            masks = masks[..., start_h:start_h + crop_h, start_w:start_w + crop_w]

        return clip, masks

    def resize(self, clip, masks):
        clip = F.interpolate(torch.FloatTensor(clip), size=(self.out_h, self.out_w), mode='bilinear')
        if masks is not None:
            masks = torch.FloatTensor(masks).unsqueeze(0).unsqueeze(0)
            masks = F.interpolate(masks, size=(self.out_h, self.out_w), mode='nearest')
            masks = masks[0][0].long()
        return clip, masks
