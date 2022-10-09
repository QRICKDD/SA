import os
from Config.attackconfig import *

print("攻击646的模型")
for pth,name in zip(RawNet646_Advset,Attacks_Name):
    print(Attacks_Name,":",len(os.listdir(pth))/1000)

print("攻击48的模型")
for pth,name in zip(RawNet48_Advset,Attacks_Name):
    print(Attacks_Name,":",len(os.listdir(pth))/1000)