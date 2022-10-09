from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.FGSM import FGSM
from Attacks.MIFGSM import MIFGSM
from Attacks.PGD import PGD

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet56.model import RawNet
import yaml

model_path = Config.RawNet56_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
yaml_path = Gconfig.RAW56_YAML_CONFIG_PATH

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

#CW
#r"F:\Adversarial-1D\AavSave\Raw56\CW"
# AdvsetPath = Config.RawNet56_CWAdvsetPath
# a = CW(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000),
#        c=10,lr=0.002,steps=100)
# a.attack()
# del(a)
AdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\CW-0003-c1"  # 0.98
a = CW(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000),
       c=1,lr=0.003,steps=100)
a.attack()
del(a)

#DeepFool
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\DeepFool-eps1e-3-myw1"  #1.0
a = DeepFool(attackdir=Config.TPATH,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
a.attack(eps=1e-3,myw=1,max_iter=50,verbose=False)
del(a)
# a = DeepFool(attackdir=Config.TPATH,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
# a.attack(eps=1e-4,myw=1,max_iter=50,verbose=False)
# del(a)

#FGSM
# a = FGSM(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
# a.attack(eps=0.015)
# del(a)
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\FGSM003" #0.177
a = FGSM(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
a.attack(eps=0.03)
del(a)
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\FGSM006"  #0.285
a = FGSM(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
a.attack(eps=0.06)
del(a)

#MIFGSM
# AdvsetPath = Config.RawNet56_MIFGSMAdvsetPath
# a = MIFGSM(model=model,attackdir=datasetPath,savedir=AdvsetPath,input_shape=(1,1,56000),
#            eps=0.03, alpha=0.001, steps=30, decay=0.5)
# a.attack()
#del(a)
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\MIFGSM003-30-1"  # 0.983
a = MIFGSM(model=model,attackdir=datasetPath,savedir=AdvsetPath,input_shape=(1,1,56000),
           eps=0.03, alpha=0.001, steps=30, decay=1)
a.attack()
del(a)
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\MIFGSM005-50-1"  # 0.997
a = MIFGSM(model=model,attackdir=datasetPath,savedir=AdvsetPath,input_shape=(1,1,56000),
           eps=0.05, alpha=0.001, steps=50, decay=1)
a.attack()
del(a)


#PGD
# AdvsetPath = Config.RawNet56_PGDAdvsetPath
# a = PGD(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
# a.attack(eps=0.05,epstep=0.001,maxiter=50)
# del(a)
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\PGD004-40-1" #ASR 969
a = PGD(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
a.attack(eps=0.04,epstep=0.001,maxiter=40)
del(a)
AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\PGD002-20-1"#ASR 0.929
a = PGD(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,56000))
a.attack(eps=0.02,epstep=0.001,maxiter=20)
del(a)