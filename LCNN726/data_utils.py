import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def genSpoof_list(dir_or_txt_path, is_eval=False):
    if (is_eval == False):
        with open(dir_or_txt_path, "r") as f:
            all_label = f.readlines()
        label_fnames = [(item.split(".")[0].strip()+'.npy') for item in all_label]
        label_label = [item.split(" ")[1].strip() for item in all_label]
        d_meta = {}
        for f, l in zip(label_fnames, label_label):
            if l == "genuine":
                l = 1
            elif l == "fake":
                l = 0
            d_meta[f] = int(l)
        return d_meta, label_fnames
    elif (is_eval):
        label_fnames = os.listdir(dir_or_txt_path)
        return label_fnames


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs    : list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.lfcc = torchaudio.transforms.LFCC(n_lfcc=60)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X = np.load(os.path.join(self.base_dir, key))
        x_inp = torch.Tensor(X)
        x_inp = self.lfcc(x_inp)
        delta = torchaudio.functional.compute_deltas(x_inp)
        delta2 = torchaudio.functional.compute_deltas(delta)
        lfccs = torch.concat([x_inp, delta, delta2], dim=0)
        lfccs = torch.transpose(lfccs, 0, 1)
        y = self.labels[key]
        return lfccs, y
