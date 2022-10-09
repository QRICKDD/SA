import numpy as np
from Config.attackconfig import Attacks_Name,RawNet56_Advset
from Attacks.Stastic_ens_MIFGSM import Static_ENS_SMIFGSM
import Tools.trans_tool as tran
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet56.model import RawNet as RawNet56
from RawNet48.model import RawNet as RawNet48
from RawNet4.model import RawNet as RawNet4
from RawNet646.model import RawNet as RawNet646
from RawNet726.model import RawNet as RawNet726
import yaml


labelPath = Config.TLABEL
datasetPath = Config.TPATH
cleandatapath = Config.TPATH
trainlabelpath = Config.TLABEL


from Tools.SNR import  cal_mean_SNR
from Config.attackconfig import *
datasetPath = Gconfig.TPATH56000

Attacks_Name=['fgsm']
RawNet56_Advset=[Config.RawNet56_FGSMAdvsetPath]
# for name,AdvsetPath in zip([Attacks_Name[-1]],[RawNet56_Advset[-1]]):
#     print(name,":",end='')
#     cal_mean_SNR(datasetPath,AdvsetPath)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model56_path = Config.RawNet56_model_2
yaml56_path = Gconfig.RAW56_YAML_CONFIG_PATH
with open(yaml56_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model56 = RawNet56(parser1['model'], device)
model56 = (model56).to(device)
model56.load_state_dict(torch.load(model56_path, map_location=device))
print('Model loaded : {}'.format(model56_path))

print("迁移性从56到56*:")
Attacks_Name=[Attacks_Name[-1]]
RawNet56_Advset=[RawNet56_Advset[-1]]
for name,advsavepath in zip(Attacks_Name,RawNet56_Advset):
    print(name, end="")
    A = tran.CalTransAtKSuccess(model56,56000,56000,advsavepath,
                                            cleandatapath,trainlabelpath,desc='RawNet')
del(model56)


model646_path = Config.RawNet646_model
yaml646_path = Gconfig.RAW646_YAML_CONFIG_PATH
with open(yaml646_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model646 = RawNet646(parser1['model'], device)
model646 = (model646).to(device)
model646.load_state_dict(torch.load(model646_path, map_location=device))
print('Model loaded : {}'.format(model646_path))

print("迁移性从56到646:")
Attacks_Name=[Attacks_Name[-1]]
RawNet56_Advset=[RawNet56_Advset[-1]]
for name,advsavepath in zip(Attacks_Name,RawNet56_Advset):
    print(name, end="")
    A = tran.CalTransAtKSuccess(model646,56000,64600,advsavepath,
                                            cleandatapath,trainlabelpath,desc='RawNet')
del(model646)

model48_path = Config.RawNet48_model
yaml48_path = Gconfig.RAW48_YAML_CONFIG_PATH
with open(yaml48_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model48 = RawNet48(parser1['model'], device)
model48 = (model48).to(device)
model48.load_state_dict(torch.load(model48_path, map_location=device))
print('Model loaded : {}'.format(model48_path))

print("迁移性从56到48:")
Attacks_Name=[Attacks_Name[-1]]
RawNet56_Advset=[RawNet56_Advset[-1]]
for name,advsavepath in zip(Attacks_Name,RawNet56_Advset):
    print(name, end="")
    A = tran.CalTransAtKSuccess(model48,56000,48000,advsavepath,
                                            cleandatapath,trainlabelpath,desc='RawNet')
del(model48)

model726_path = Config.RawNet726_model
yaml726_path = Gconfig.RAW726_YAML_CONFIG_PATH
with open(yaml726_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model726 = RawNet726(parser1['model'], device)
model726 = (model726).to(device)
model726.load_state_dict(torch.load(model726_path, map_location=device))
print('Model loaded : {}'.format(model726_path))

print("迁移性从56到726:")
Attacks_Name=[Attacks_Name[-1]]
RawNet56_Advset=[RawNet56_Advset[-1]]
for name,advsavepath in zip(Attacks_Name,RawNet56_Advset):
    print(name, end="")
    A = tran.CalTransAtKSuccess(model726,56000,72600,advsavepath,
                                            cleandatapath,trainlabelpath,desc='RawNet')
del(model726)

model4_path = Config.RawNet4_model
yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
with open(yaml4_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model4 = RawNet4(parser1['model'], device)
model4 = (model4).to(device)
model4.load_state_dict(torch.load(model4_path, map_location=device))
print('Model loaded : {}'.format(model4_path))

print("迁移性从56到4:")
Attacks_Name=[Attacks_Name[-1]]
RawNet56_Advset=[RawNet56_Advset[-1]]
for name,advsavepath in zip(Attacks_Name,RawNet56_Advset):
    print(name, end="")
    A = tran.CalTransAtKSuccess(model4,56000,40000,advsavepath,
                                            cleandatapath,trainlabelpath,desc='RawNet')
del(model4)
