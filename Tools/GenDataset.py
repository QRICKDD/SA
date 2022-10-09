from Tools.RName import flctonpy,npytoflc
import Config.globalconfig as Gconfig
import os
import soundfile as sf
import numpy as np
import shutil
import tqdm
def handleSourceTraget(source_length,target_length,perturb):
    if source_length>target_length:
        return perturb[:target_length]
    else:
        return np.concatenate((perturb,perturb[:int(target_length-source_length)]),axis=0)

def Gendataset(old_dataset,new_dataset,old_label_path,new_label_path,source_length,target_length):
    if os.path.exists(new_dataset)==False:
        os.makedirs(new_dataset)
        print("create new director: ",new_dataset)
    shutil.copy(old_label_path,new_label_path)
    all_dataset=os.listdir(old_dataset)
    for item in tqdm.tqdm(all_dataset):
        x,sr=sf.read(os.path.join(old_dataset,item))
        x=handleSourceTraget(source_length,target_length,x)
        np.save(os.path.join(new_dataset,flctonpy(item)),x)

old_dataset=Gconfig.TPATH64600
new_dataset=Gconfig.TPATH56000
old_label=Gconfig.TLABEL64600
new_label=Gconfig.TLABEL56000

Gendataset(old_dataset=old_dataset,new_dataset=new_dataset,
           old_label_path=old_label,new_label_path=new_label,
           source_length=64600,target_length=56000)

old_dataset=Gconfig.EPATH64600
new_dataset=Gconfig.EPATH56000
old_label=Gconfig.ELABEL64600
new_label=Gconfig.ELABEL56000

Gendataset(old_dataset=old_dataset,new_dataset=new_dataset,
           old_label_path=old_label,new_label_path=new_label,
           source_length=64600,target_length=56000)