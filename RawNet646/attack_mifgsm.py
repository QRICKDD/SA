from Attacks.MIFGSM import MIFGSM
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet646.model import RawNet
import yaml

model_path = Config.RawNet646_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.RawNet646_MIFGSMAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yaml_path = Gconfig.RAW_YAML_CONFIG_PATH
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

a = MIFGSM(model=model,attackdir=datasetPath,savedir=AdvsetPath,input_shape=(1,1,64600),
           eps=0.03, alpha=0.001, steps=30, decay=0.5)
a.attack()
