from Tools.SNR import  cal_mean_SNR
import Config.attackconfig as Config
import Config.globalconfig as GConfig
from Config.attackconfig import *
datasetPath = GConfig.TPATH56000

Attacks_Name=["xxx"]
RawNet56_Advset=[r"F:\Adversarial-1D\AavSave\LCNN56\FGSM"]
for name,AdvsetPath in zip(Attacks_Name,RawNet56_Advset):
    print(name,":",end='')
    cal_mean_SNR(datasetPath,AdvsetPath)