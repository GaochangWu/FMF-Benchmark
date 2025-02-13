import torch
from torch.utils.data import Dataset
from FMF.utils.augment import HorizontalFlip

import os
import json
import h5py
import numpy as np
import random
import pandas as pd


class VideoDataset(Dataset):
    def __init__(self, mode, prefix, num_frames=10, sampling_rate=1, task='classification'):
        assert task in ['classification', 'segmentation']
        self.task = task
        self.mode = mode if mode != 'val' else 'test'
        if self.mode == 'test':
            random.seed(0)
        self.prefix = prefix
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate

        fmf_files = self.get_fmf_files(prefix)
        self.fmf_files = fmf_files

        # for train, len(frame_split[0, 1, 2]) = 20826, 73437, 120658
        # for test,  len(frame_split[0, 1, 2]) = 10000, 15000, 25000
        self.frame_split = self.get_frame_split(mode, prefix, fmf_files)

        # len(labels[0, 1, 2]) = 31826, 89937, 148158
        self.videos, self.labels = self.get_videos_labels(prefix, fmf_files, task)
        # len(timestamps[0, 1, 2]) = 31826, 89937, 148158
        self.video_times = self.get_timestamps(prefix, fmf_files)

        self.lens = [len(fi) for fi in self.frame_split]

        self.Flip = HorizontalFlip(p=0.5)

    def __getitem__(self, item):
        video_idx = 0
        frame_idx = 0
        for ln in self.lens:
            if item < ln:
                frame_idx = self.frame_split[video_idx][item]
                if frame_idx < self.num_frames * self.sampling_rate - 1:
                    frame_idx = random.randint(self.num_frames * self.sampling_rate - 1, ln - 1)
                break
            item -= ln
            video_idx += 1

        # get video clip and label
        idx_for_clip = np.arange(frame_idx - (self.num_frames - 1) * self.sampling_rate, frame_idx + 1,
                                 self.sampling_rate)
        idx_for_clip = np.maximum(idx_for_clip, 0)
        clip = self.videos[video_idx][idx_for_clip]
        clip = clip.transpose(1, 0, 2, 3)  # T C .. -> C T ..
        label = self.labels[video_idx][frame_idx]
        clip = clip.astype(np.float32)
        label = label.astype(np.int64)

        # data augmentation
        if self.mode == 'train':
            clip, label = self.Flip(clip, label)

        return clip, label

    def __len__(self):
        return sum(self.lens)

    def __repr__(self):
        msg = (f'{self.__class__.__name__}: \n'
               f'task={self.task}, \n'
               f'mode={self.mode}, \n'
               f'samples={self.__len__()}, \n'
               f'prefix={self.prefix}, \n'
               f'num_frames={self.num_frames}, \n'
               f'sampling_rate={self.sampling_rate}')
        return msg

    @staticmethod
    def get_fmf_files(prefix):
        filenames = [fn.split('.')[0] for fn in os.listdir(os.path.join(prefix, 'videos'))]
        video_files = [fn + '.mat' for fn in filenames]
        current_files = [fn + '.csv' for fn in filenames]
        timestamp_files = [fn + '.csv' for fn in filenames]
        fmf_files = {
            'videos': video_files,
            'currents': current_files,
            'timestamps': timestamp_files
        }
        return fmf_files

    @staticmethod
    def get_frame_split(mode, prefix, fmf_files):
        """
        for train, len(frame_indexes[0, 1, 2]) = 20826, 73437, 120658
        for test,  len(frame_indexes[0, 1, 2]) = 10000, 15000, 25000
        """
        furnaces = fmf_files['videos']
        split_file = os.path.join(prefix, mode + '.json')
        with open(split_file, 'r') as f:
            furnace2indexes = json.load(f)
        frame_split = [furnace2indexes[fur] for fur in furnaces]
        return frame_split

    @staticmethod
    def get_videos_labels(prefix, fmf_files, task):
        """
        len(labels[0, 1, 2]) = 31826, 89937, 148158
        """
        videos = []
        labels = []
        video_files = fmf_files['videos']
        for fn in video_files:
            fp = os.path.join(prefix, 'videos', fn)
            with h5py.File(fp, 'r') as reader:
                video = np.array(reader.get('data'), dtype=np.uint8)
                label = np.array(reader.get('label'), dtype=np.uint8)
                if task == 'classification':
                    label = np.max(label, axis=(1, 2))
                label[label > 0] = 1
                labels.append(label)  # T W H
                videos.append(video)  # T C W H
        return videos, labels

    @staticmethod
    def get_timestamps(prefix, fmf_files):
        """
        len(timestamps[0, 1, 2]) = 31826, 89937, 148158
        """
        timestamps = []
        timestamp_files = fmf_files['timestamps']
        for fn in timestamp_files:
            fp = os.path.join(prefix, 'timestamps', fn)
            df = pd.read_csv(fp)
            ts = list(df.iloc[:, 0])
            timestamps.append(ts)
        return timestamps


def main():
    mode = 'train'
    prefix = r'D:\datasets\FMF-Benchmark\pixel-level'

    video_dataset = VideoDataset(mode, prefix, num_frames=10, sampling_rate=1, task='segmentation')
    print(video_dataset)

    clip, label = video_dataset[0]
    print(clip.shape, label.shape)


if __name__ == '__main__':
    main()
