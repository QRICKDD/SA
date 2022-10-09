from Attacks.PGD_ens_2 import PGD_ens_2

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet56.model import RawNet as RawNet56
from RawNet4.model import RawNet as RawNet4
from RawNet4.model import RawNet as RawNet48
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

# model56_path2 = Config.RawNet56_model_2
# with open(yaml56_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model56_2 = RawNet56(parser1['model'], device)
# model56_2 = (model56_2).to(device)
# model56_2.load_state_dict(torch.load(model56_path2, map_location=device))
# print('Model loaded : {}'.format(model56_path2))

model4_path = Config.RawNet4_model
yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
with open(yaml4_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model4 = RawNet4(parser1['model'], device)
model4 = (model4).to(device)
model4.load_state_dict(torch.load(model4_path, map_location=device))
print('Model loaded : {}'.format(model4_path))

# model48_path = Config.RawNet48_model
# yaml48_path = Gconfig.RAW48_YAML_CONFIG_PATH
# with open(yaml48_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model48 = RawNet48(parser1['model'], device)
# model48 = (model48).to(device)
# model48.load_state_dict(torch.load(model48_path, map_location=device))
# print('Model loaded : {}'.format(model48_path))

#AdvsetPath = Config.RawNet56_Subens_CWAdvsetPath
AdvsetPath =r"F:\Adversarial-1D\AavSave\Raw56\PGD_sub_56_4_decay1"
a = PGD_ens_2(
    attackdir=datasetPath,savedir=AdvsetPath,
    model=model56,model_few=model4,
    sub_number=5,v_num=3,v_range=(-0.01,0.01),
    input_shape=(1, 1,56000),input_shape_few=(1,1,40000),
    eps=0.05,alpha=0.001,steps=60,decay=1.0)

a.attack()
del(a)
