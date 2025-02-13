import torch
from torch.utils.data import Dataset

import os
import json
import h5py
import numpy as np
import pandas as pd


class CurrentDataset(Dataset):
    def __init__(self, mode, prefix, current_seconds=10):
        self.mode = mode
        self.prefix = prefix
        self.current_seconds = current_seconds

        fmf_files = self.get_fmf_files(prefix)
        self.fmf_files = fmf_files

        # for train, len(frame_indexes[0, 1, 2]) = 20826, 73437, 120658
        # for test,  len(frame_indexes[0, 1, 2]) = 10000, 15000, 25000
        self.frame_split = self.get_frame_split(mode, prefix, fmf_files)
        self.video_times = self.get_timestamps(prefix, fmf_files)

        # len(currents[0, 1, 2]) = 1298, 3598, 5993
        # currents[2]["2023-05-10 21:33:27"] = [[6343.75, 8157.79, 6149.69], [11369.2, 13860, 12341.4]]
        self.currents, self.cur_times = self.get_currents(prefix, fmf_files)

        # len(labels[0, 1, 2]) = 31826, 89937, 148158
        self.frame_labels = self.get_frame_labels(prefix, fmf_files)

        # for train, len(datas/labels[0, 1, 2]) = 835, 2940, 4826
        # for test,  len(datas/labels[0, 1, 2]) = 402, 602, 1000
        self.datas, self.labels = self.get_datas_and_labels()

        self.lens = [len(lb) for lb in self.labels]

    def __getitem__(self, item):
        video_idx = 0
        end_idx = 0
        label = None
        for ln in self.lens:
            if item < ln:
                end_sample = self.datas[video_idx][item]
                label = self.labels[video_idx][item]
                end_idx = self.cur_times[video_idx].index(end_sample)
                break
            item -= ln
            video_idx += 1

        # get currents
        sampled_times = self.cur_times[video_idx][max(end_idx + 1 - self.current_seconds, 0): end_idx + 1]
        x = []
        for ts in sampled_times:
            x += self.currents[video_idx][ts]
        x = np.array(x, dtype=np.float32)

        # If the length of the current data is not sufficient, interpolate it
        if x.shape[0] < 2 * self.current_seconds:
            expand_indexes = np.linspace(start=0, stop=x.shape[0] - 1, num=2 * self.current_seconds).astype(np.int32)
            x = x[expand_indexes]

        return x, label

    def __len__(self):
        # for train, len = 8601
        # for test, len = 2004
        return sum(self.lens)

    def __repr__(self):
        msg = (f'{self.__class__.__name__}: \n'
               f'mode={self.mode}, \n'
               f'samples={self.__len__()}, \n'
               f'prefix={self.prefix}, \n'
               f'current_seconds={self.current_seconds}')
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
    def get_frame_labels(prefix, fmf_files):
        """
        len(labels[0, 1, 2]) = 31826, 89937, 148158
        """
        labels = []
        furnace_files = [os.path.join(prefix, 'videos', filename) for filename in fmf_files['videos']]
        for ff in furnace_files:
            with h5py.File(ff, 'r') as reader:
                label = np.array(reader.get('label'))
                label = np.max(label, axis=(1, 2))
                label[label > 0] = 1
                labels.append(label)
        return labels

    def get_datas_and_labels(self):
        datas, labels = [], []

        for fi, ts in zip(self.frame_split, self.video_times):
            fi, ts = np.array(fi), np.array(ts)
            ts = ts[fi].tolist()  # Select timestamps based on frame split
            data = list(set(ts))  # Remove duplicate timestamps
            data.sort(key=ts.index)  # Ensure that the timestamp order remains unchanged
            datas.append(data)

        for dt, fl, ts in zip(datas, self.frame_labels, self.video_times):
            label = []
            for timestamp in dt:
                idx = ts.index(timestamp)
                summed = np.sum(fl[idx:idx + 25])
                if summed > 0:
                    label.append(1)
                else:
                    label.append(0)
            labels.append(label)

        return datas, labels

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

    @staticmethod
    def remove_duplication(lst):
        new_lst = list(set(lst))
        new_lst.sort(key=lst.index)
        return new_lst


def main():
    mode = 'test'
    prefix = r'D:\datasets\FMF-Benchmark\pixel-level'

    current_dataset = CurrentDataset(mode, prefix, current_seconds=120)
    print(current_dataset)

    x, y = current_dataset[0]
    print(x.shape, y)


if __name__ == '__main__':
    main()
