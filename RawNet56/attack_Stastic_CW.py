from Attacks.CW_Stastic import CW_Stastic

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet56.model import RawNet as RawNet56
from RawNet4.model import RawNet as RawNet4
from RawNet726.model import RawNet as RawNet726
import yaml

labelPath = Config.TLABEL
datasetPath = Config.TPATH
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model56_path = Config.RawNet56_model
yaml56_path = Gconfig.RAW56_YAML_CONFIG_PATH
with open(yaml56_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model56 = RawNet56(parser1['model'], device)
model56 = (model56).to(device)
model56.load_state_dict(torch.load(model56_path, map_location=device))
print('Model loaded : {}'.format(model56_path))


model726_path = Config.RawNet726_model
yaml726_path = Gconfig.RAW726_YAML_CONFIG_PATH
with open(yaml726_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model726 = RawNet726(parser1['model'], device)
model726 = (model726).to(device)
model726.load_state_dict(torch.load(model726_path, map_location=device))
print('Model loaded : {}'.format(model726_path))

model4_path = Config.RawNet4_model
yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
with open(yaml4_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model4 = RawNet4(parser1['model'], device)
model4 = (model4).to(device)
model4.load_state_dict(torch.load(model4_path, map_location=device))
print('Model loaded : {}'.format(model4_path))


AdvsetPath = Config.RawNet56_Satics_CWAdvsetPath
a = CW_Stastic(
    attackdir=datasetPath,savedir=AdvsetPath,
    model=model56,model_few=model4,model_more=model726,
    input_shape=(1, 1,56000),input_shape_few=(1,1,40000),input_shape_more=(1,1,72600),
    c=10,c_few=10,c_more=10,lr=0.0001,steps=100)

a.attack()
del(a)
