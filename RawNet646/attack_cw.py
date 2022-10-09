from Attacks.CW import CW
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet646.model import RawNet
import yaml
import matplotlib.pyplot as plt
from Tools.SNR import SNR_singlech
model_path = Config.RawNet646_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.RawNet646_CWAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yaml_path = Gconfig.RAW_YAML_CONFIG_PATH
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

a = CW(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,64600),
       c=10,lr=0.002,steps=100)
a.attack()

