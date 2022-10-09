import os
import soundfile as sf
import numpy as np
from Tools.RName import flctonpy
from Tools.RName import npytoflc

# 计算信噪比
def SNR_singlech(clean_wav, noisy_wav):
    est_noise = noisy_wav-clean_wav
    if max(clean_wav)>1:
        print("原始音频剪切问题---")
    if max(noisy_wav)>1:
        print("对抗样本生产产生问题")
    # 计算信噪比
    SNR = 10 * np.log10((np.sum(clean_wav ** 2)) / (np.sum(est_noise ** 2)))
    return SNR

def cal_mean_SNR(attack_dir,advsae_dir):
    alladvname=os.listdir(advsae_dir)
    allSNR=0
    for item in alladvname:
        prewav= np.load(os.path.join(advsae_dir, flctonpy(item)),allow_pickle=True)
        #cleanwav, sr = sf.read(os.path.join(attack_dir, npytoflc(item)))
        cleanwav= np.load(os.path.join(attack_dir, flctonpy(item)),allow_pickle=True)
        snr=SNR_singlech(clean_wav=cleanwav,noisy_wav=prewav)
        allSNR+=snr
    print("平均信噪比 为:",str(allSNR/len(alladvname)))
