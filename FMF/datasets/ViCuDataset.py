import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import json
import h5py
import pandas as pd
import numpy as np
import random
from collections import OrderedDict
from FMF.utils.augment import HorizontalFlip


class ViCuDataset(Dataset):
    def __init__(self, mode, prefix, num_frames=8, sampling_rate=1, current_seconds=120, task='classification'):
        assert task in ['classification', 'segmentation']
        self.task = task
        self.mode = mode if mode != 'val' else 'test'
        if self.mode == 'test':
            random.seed(0)
        self.prefix = prefix
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.current_seconds = current_seconds

        fmf_files = self.get_fmf_files(prefix)
        self.fmf_files = fmf_files

        # for train, len(frame_split[0, 1, 2]) = 20826, 73437, 120658
        # for test,  len(frame_split[0, 1, 2]) = 10000, 15000, 25000
        self.frame_split = self.get_frame_split(mode, prefix, fmf_files)

        # len(currents[0, 1, 2]) = 1298, 3598, 5993
        # currents[2]["2023-05-10 21:33:27"] = [[6343.75, 8157.79, 6149.69], [11369.2, 13860, 12341.4]]
        self.currents, self.cur_times = self.get_currents(prefix, fmf_files)

        # len(labels[0, 1, 2]) = 31826, 89937, 148158,
        self.videos, self.labels = self.get_videos_labels(prefix, fmf_files, task)
        # len(timestamps[0, 1, 2]) = 31826, 89937, 148158
        self.video_times = self.get_timestamps(prefix, fmf_files)

        self.lens = [len(fi) for fi in self.frame_split]

        self.Flip = HorizontalFlip(0.5)

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
        idx_for_clip = np.arange(frame_idx - (self.num_frames - 1) * self.sampling_rate, frame_idx + 1, self.sampling_rate)
        idx_for_clip = np.maximum(idx_for_clip, 0)
        clip = self.videos[video_idx][idx_for_clip]
        clip = clip.transpose(1, 0, 2, 3)  # T C .. -> C T ..
        label = self.labels[video_idx][frame_idx]
        clip = clip.astype(np.float32)
        label = label.astype(np.int64)

        # get currents
        time_sampled = self.video_times[video_idx][frame_idx]
        t_idx_in_cur = self.cur_times[video_idx].index(time_sampled)
        sampled_times = self.cur_times[video_idx][max(t_idx_in_cur + 1 - self.current_seconds, 0):t_idx_in_cur + 1]
        currents = []
        for ts in sampled_times:
            currents += self.currents[video_idx][ts]
        currents = np.array(currents, dtype=np.float32)
        if currents.shape[0] < 2 * self.current_seconds:
            expand_indexes = np.linspace(start=0, stop=currents.shape[0] - 1, num=2 * self.current_seconds).astype(
                np.int32)
            currents = currents[expand_indexes]

        # data augmentation
        if self.mode == 'train':
            clip, label = self.Flip(clip, label)

        return clip, currents, label

    def __len__(self):
        return sum(self.lens)

    def __repr__(self):
        msg = (f'{self.__class__.__name__}: \n'
               f'task={self.task}, \n'
               f'mode={self.mode}, \n'
               f'samples={self.__len__()}, \n'
               f'prefix={self.prefix}, \n'
               f'num_frames={self.num_frames}, \n'
               f'sampling_rate={self.sampling_rate}, \n'
               f'current_seconds={self.current_seconds}')
        return msg

    @staticmethod
    def get_fmf_files(prefix):
        filenames = [fn.split('.')[0] for fn in os.listdir(os.path.join(prefix, 'videos'))]
        video_files = [fn+'.mat' for fn in filenames]
        current_files = [fn+'.csv' for fn in filenames]
        timestamp_files = [fn + '.csv' for fn in filenames]
        fmf_files = {
            'videos': video_files,
            'currents': current_files,
            'timestamps': timestamp_files
        }
        return fmf_files

    @staticmethod
    def get_currents(prefix, fmf_files):
        """
        len(currents[0, 1, 2]) = 1298, 3598, 5993
        currents[2]["2023-05-10 21:33:27"] = [[6343.75, 8157.79, 6149.69], [11369.2, 13860, 12341.4]]
        """
        currents = []
        cur_times = []
        current_files = fmf_files['currents']

        for fn in current_files:
            current = dict()
            cur_time = []
            fp = os.path.join(prefix, 'currents', fn)
            df = pd.read_csv(fp)

            for i in range(0, df.shape[0], 2):
                ts = df.iloc[i, 0]  # time
                cur1 = list(df.iloc[i, 1:])  # [a1, b1, c1]
                cur2 = list(df.iloc[i + 1, 1:])  # [a2, b2, c2]
                current[ts] = [cur1, cur2]
                cur_time.append(ts)

            currents.append(current)
            cur_times.append(cur_time)

        return currents, cur_times

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

    vicu_dataset = ViCuDataset(mode, prefix, num_frames=10, sampling_rate=1, current_seconds=120, task='classification')
    print(vicu_dataset)

    clip, current, label = vicu_dataset[0]
    print(clip.shape, current.shape, label.shape)
    print(label, type(label))


if __name__ == '__main__':
    main()
