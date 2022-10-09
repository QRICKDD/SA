import os
from torch import Tensor
import soundfile as sf
from torch.utils.data import Dataset
import numpy as np
from Tools.RName import flctonpy,npytoflc
def genSpoof_list(dir_or_txt_path, is_eval=False):
    if (is_eval==False):
        with open(dir_or_txt_path, "r") as f:
            all_label = f.readlines()
        label_fnames = [item.split(" ")[0] for item in all_label]
        label_label = [item.split(" ")[1].strip() for item in all_label]
        d_meta = {}
        for f,l in zip(label_fnames,label_label):
            if l=="genuine":
                l=1
            elif l=="fake":
                l=0
            d_meta[f]=int(l)
        return d_meta, label_fnames
    elif (is_eval):
        label_fnames=os.listdir(dir_or_txt_path)
        return label_fnames


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs    : list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        #X, fs = sf.read(os.path.join(self.base_dir,key))
        X=np.load(os.path.join(self.base_dir,flctonpy(key)))
        #X.reshape(1,-1)
        #归一化
        #X = librosa.util.normalize(X)
        x_inp = Tensor(X)
        # x = x.reshape(1, -1)
        # x = self.lfcc(x)
        y = self.labels[key]
        return x_inp, y
