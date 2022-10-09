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



AdvsetPath=r"F:\Adversarial-1D\AavSave\Raw56\MIFGSM003-20-1"#0.9292
a = MIFGSM(model=model,attackdir=datasetPath,savedir=AdvsetPath,input_shape=(1,1,56000),
           eps=0.02, alpha=0.001, steps=20, decay=1)
a.attack()
del(a)


