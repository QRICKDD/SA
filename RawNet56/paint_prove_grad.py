import torch
import yaml

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
from Prove.paint_pgd_ens import paint_PGD_ens
from Prove.paint_pgd_ens3 import paint_PGD_ens3
from RawNet4.model import RawNet as RawNet4
from RawNet56.model import RawNet as RawNet56
from RawNet726.model import RawNet as RawNet726

labelPath = Config.TLABEL
datasetPath = Config.TPATH
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model4_path = Config.RawNet4_model
yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
with open(yaml4_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model4 = RawNet4(parser1['model'], device)
model4 = (model4).to(device)
model4.load_state_dict(torch.load(model4_path, map_location=device))
print('Model loaded : {}'.format(model4_path))

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


def PGD_double(model1, model2, shape1, shape2):
    AdvsetPath = r"F:\Adversarial-1D\AavSave\RawNet56\TempPaint_" + str(shape1) + "_" + str(shape2)
    Npsave = r"F:\Adversarial-1D\AavSave\RawNet56\PGD_" + str(shape1) + "_" + str(shape2)
    print(AdvsetPath)
    print(Npsave)
    a = paint_PGD_ens(
        attackdir=datasetPath, savedir=AdvsetPath, grad_save_path=Npsave,
        model=model1, model_few=model2,
        sub_number=0, v_num=0, v_range=(-0.01, 0.01),
        input_shape=(1, 1, shape1), input_shape_few=(1, 1, shape2),
        eps=0.05, alpha=0.001, steps=150, is_rawnet=True)
    a.attack()
    del (a)


def PGD_triple():
    AdvsetPath = r"F:\Adversarial-1D\AavSave\RawNet56\TempPaint_4_56_726"
    Npsave = r"F:\Adversarial-1D\AavSave\RawNet56\PGD_4_56_726"
    print(AdvsetPath)
    print(Npsave)
    a = paint_PGD_ens3(
        attackdir=datasetPath, savedir=AdvsetPath, grad_save_path=Npsave,
        model=model56, model_few=model4, model_more=model726,
        input_shape=(1, 1, 56000), input_shape_more=(1, 1, 72600),
        input_shape_few=(1, 1, 40000),
        eps=0.05, alpha=0.001, steps=50, is_rawnet=True)
    a.attack()
    del (a)


PGD_double(model56, model4, 56000, 40000)
PGD_double(model56, model726, 56000, 72600)
PGD_triple()

# model56_path2 = Config.RawNet56_model_2
# with open(yaml56_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model56_2 = RawNet56(parser1['model'], device)
# model56_2 = (model56_2).to(device)
# model56_2.load_state_dict(torch.load(model56_path2, map_location=device))
# print('Model loaded : {}'.format(model56_path2))
#
# model4_path = Config.RawNet4_model
# yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
# with open(yaml4_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model4 = RawNet4(parser1['model'], device)
# model4 = (model4).to(device)
# model4.load_state_dict(torch.load(model4_path, map_location=device))
# print('Model loaded : {}'.format(model4_path))

# model48_path = Config.RawNet48_model
# yaml48_path = Gconfig.RAW48_YAML_CONFIG_PATH
# with open(yaml48_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model48 = RawNet48(parser1['model'], device)
# model48 = (model48).to(device)
# model48.load_state_dict(torch.load(model48_path, map_location=device))
# print('Model loaded : {}'.format(model48_path))
#
# model726_path = Config.RawNet726_model
# yaml726_path = Gconfig.RAW726_YAML_CONFIG_PATH
# with open(yaml726_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model726 = RawNet726(parser1['model'], device)
# model726 = (model726).to(device)
# model726.load_state_dict(torch.load(model726_path, map_location=device))
# print('Model loaded : {}'.format(model726_path))

# model646_path = Config.RawNet646_model
# yaml646_path = Gconfig.RAW646_YAML_CONFIG_PATH
# with open(yaml646_path, 'r') as f_yaml:
#     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
# model646 = RawNet646(parser1['model'], device)
# model646 = (model646).to(device)
# model646.load_state_dict(torch.load(model646_path, map_location=device))
# print('Model loaded : {}'.format(model646_path))

# AdvsetPath = Config.RawNet56_Subens_CWAdvsetPath
# AdvsetPath =r"F:\Adversarial-1D\AavSave\Raw56\temp_paint\temp"
# Npsave=r"F:\Adversarial-1D\Paint_Adv2\56_726"
# a = paint_PGD_ens(
#     attackdir=datasetPath,savedir=AdvsetPath,grad_save_path=Npsave,
#     model=model56,model_few=model726,
#     sub_number=0,v_num=0,v_range=(-0.01,0.01),
#     input_shape=(1, 1,56000),input_shape_few=(1,1,72600),
#     eps=0.05,alpha=0.001,steps=150)
#
# a.attack()
# del(a)
