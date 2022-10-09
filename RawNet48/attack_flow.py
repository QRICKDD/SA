from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.FGSM import FGSM
from Attacks.MIFGSM import MIFGSM
from Attacks.PGD import PGD

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
import yaml

from RawNet48.model import RawNet
model_path = Config.RawNet48_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
yaml_path = Gconfig.RAW48_YAML_CONFIG_PATH

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))


#CW
AdvsetPath = Config.RawNet48_CWAdvsetPath
a = CW(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,64600),
       c=10,lr=0.002,steps=100)
a.attack()
del(a)
#DeepFool
AdvsetPath=Config.RawNet48_DeepfoolAdvsetPath
a = DeepFool(attackdir=Config.TPATH,savedir=AdvsetPath,model=model,
             input_shape=(1,1,64600))
a.attack(eps=1e-4,myw=0.8,max_iter=50,verbose=False)
del(a)
#FGSM
AdvsetPath=Config.RawNet48_FGSMAdvsetPath
a = FGSM(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,64600))
a.attack(eps=0.06)
del(a)
#MIFGSM
AdvsetPath = Config.RawNet48_MIFGSMAdvsetPath
a = MIFGSM(model=model,attackdir=datasetPath,savedir=AdvsetPath,input_shape=(1,1,64600),
           eps=0.03, alpha=0.001, steps=30, decay=0.5)
a.attack()
del(a)
#PGD
AdvsetPath = Config.RawNet48_PGDAdvsetPath
a = PGD(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,64600))
a.attack(eps=0.05,epstep=0.001,maxiter=50)
del(a)