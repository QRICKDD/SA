import torch
import yaml

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
from Attacks.SA_1D_3 import SA_1D_ens3
from RawNet4.model import RawNet as RawNet4
from RawNet56.model import RawNet as RawNet56
from RawNet726.model import RawNet as RawNet726

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

model4_path = Config.RawNet4_model
yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
with open(yaml4_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model4 = RawNet4(parser1['model'], device)
model4 = (model4).to(device)
model4.load_state_dict(torch.load(model4_path, map_location=device))
print('Model loaded : {}'.format(model4_path))

model726_path = Config.RawNet726_model
yaml726_path = Gconfig.RAW726_YAML_CONFIG_PATH
with open(yaml726_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model726 = RawNet726(parser1['model'], device)
model726 = (model726).to(device)
model726.load_state_dict(torch.load(model726_path, map_location=device))
print('Model loaded : {}'.format(model726_path))


def SA_triple(sub_number, v_num):
    AdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn{}-v{}".format(
        sub_number, v_num)
    print(AdvsetPath)
    a = SA_1D_ens3(
        attackdir=datasetPath, savedir=AdvsetPath,
        model=model56, model_few=model4, model_more=model726,
        input_shape=(1, 1, 56000), input_shape_few=(1, 1, 40000), input_shape_more=(1, 1, 72600),
        c_norm=1, c_norm_more=1.0, c_norm_few=1.0, c_more=1, c_few=1, gw=False,
        sub_number=sub_number,
        v_num=v_num, v_range=(-0.01, 0.01),
        eps=0.05, alpha=0.001, steps=50, decay=0.1, is_rawnet=True)
    a.attack()
    del (a)


if __name__ == "__main__":
    SA_triple(sub_number=0, v_num=0)
    SA_triple(sub_number=0, v_num=3)
    SA_triple(sub_number=5, v_num=0)
    SA_triple(sub_number=5, v_num=3)
    SA_triple(sub_number=5, v_num=5)
    SA_triple(sub_number=10, v_num=3)
    SA_triple(sub_number=10, v_num=5)
    SA_triple(sub_number=20, v_num=0)
    SA_triple(sub_number=20, v_num=3)
    SA_triple(sub_number=20, v_num=5)
