from Tools.SNR import  cal_mean_SNR
import Config.attackconfig as Config
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
from Tools.RName import flctonpy,npytoflc
datasetPath = Config.TPATH
AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\FGSM"
all_fname=os.listdir(AdvsetPath)
for fname in all_fname:
    x1,sr=sf.read(os.path.join(datasetPath,npytoflc(fname)))
    x2= np.load(os.path.join(AdvsetPath, flctonpy(fname)),allow_pickle=True)
    ss2=x2.shape[0]
    x1=x1[:ss2]
    res = x2 - x1

    plt.plot(x2)
    plt.plot(res)
    plt.show()
    plt.plot(x2)
    plt.plot(x1)
    plt.show()

    print(max(res),"==",min(res))
