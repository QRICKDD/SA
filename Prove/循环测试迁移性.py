import torch
import yaml

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import Tools.trans_tool as tran
from RawNet48.model import RawNet as RawNet48
from RawNet56.model import RawNet as RawNet56
from RawNet646.model import RawNet as RawNet646

labelPath = Config.TLABEL
datasetPath = Config.TPATH
cleandatapath = Config.TPATH
trainlabelpath = Config.TLABEL

from Tools.SNR import cal_mean_SNR

datasetPath = Gconfig.TPATH56000

Attacks_Path = [
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn0-v0-step50-decay0-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn0-v3-step50-decay0-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn5-v0-step50-decay0-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn5-v3-step50-decay0",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn5-v3-step50-decay0-nogw",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn5-v3-step50-decay0.1-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn10-v3-step50-decay0-ngw-True",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn10-v3-step50-decay0.1-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn20-v0-step50-decay0-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn20-v0-step50-decay0-ngw-True",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn20-v5-step50-decay0.0-ngw-False",
    r"F:\Adversarial-1D\AavSave\Raw56\SA-1D\56-726-4-sn20-v5-step50-decay0.1-ngw-False",
]

for AdvsetPath in Attacks_Path:
    print(AdvsetPath, ":", end='')
    cal_mean_SNR(datasetPath, AdvsetPath)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 遍历测试迁移性
for my_adv_set_item in Attacks_Path:
    print('开始测试，路径如下')
    print(my_adv_set_item)
    # #测4的迁移性
    # model4_path = Config.RawNet4_model
    # yaml4_path = Gconfig.RAW4_YAML_CONFIG_PATH
    # with open(yaml4_path, 'r') as f_yaml:
    #     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    # model4 = RawNet4(parser1['model'], device)
    # model4 = (model4).to(device)
    # model4.load_state_dict(torch.load(model4_path, map_location=device))
    # print('Model loaded : {}'.format(model4_path))
    #
    # print("迁移性从56到4:")
    # A = tran.CalTransAtKSuccess(model4, 56000, 40000, my_adv_set_item,
    #                                 cleandatapath, trainlabelpath, desc='RawNet')
    # del (model4)

    # 测48的迁移性
    model48_path = Config.RawNet48_model
    yaml48_path = Gconfig.RAW48_YAML_CONFIG_PATH
    with open(yaml48_path, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    model48 = RawNet48(parser1['model'], device)
    model48 = (model48).to(device)
    model48.load_state_dict(torch.load(model48_path, map_location=device))
    print('Model loaded : {}'.format(model48_path))

    print("迁移性从56到48:")

    A = tran.CalTransAtKSuccess(model48, 56000, 48000, my_adv_set_item,
                                cleandatapath, trainlabelpath, desc='RawNet')
    del (model48)

    # 测56的迁移性
    model56_path = Config.RawNet56_model_2
    yaml56_path = Gconfig.RAW56_YAML_CONFIG_PATH
    with open(yaml56_path, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    model56 = RawNet56(parser1['model'], device)
    model56 = (model56).to(device)
    model56.load_state_dict(torch.load(model56_path, map_location=device))
    print('Model loaded : {}'.format(model56_path))

    print("迁移性从56到56*:")

    A = tran.CalTransAtKSuccess(model56, 56000, 56000, my_adv_set_item,
                                cleandatapath, trainlabelpath, desc='RawNet')
    del (model56)

    # 测646的迁移性
    model646_path = Config.RawNet646_model
    yaml646_path = Gconfig.RAW646_YAML_CONFIG_PATH
    with open(yaml646_path, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    model646 = RawNet646(parser1['model'], device)
    model646 = (model646).to(device)
    model646.load_state_dict(torch.load(model646_path, map_location=device))
    print('Model loaded : {}'.format(model646_path))

    print("迁移性从56到646:")

    A = tran.CalTransAtKSuccess(model646, 56000, 64600, my_adv_set_item,
                                cleandatapath, trainlabelpath, desc='RawNet')
    del (model646)

    # 测726的迁移性
    # model726_path = Config.RawNet726_model
    # yaml726_path = Gconfig.RAW726_YAML_CONFIG_PATH
    # with open(yaml726_path, 'r') as f_yaml:
    #     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    # model726 = RawNet726(parser1['model'], device)
    # model726 = (model726).to(device)
    # model726.load_state_dict(torch.load(model726_path, map_location=device))
    # print('Model loaded : {}'.format(model726_path))
    #
    # print("迁移性从56到726:")
    #
    # A = tran.CalTransAtKSuccess(model726,56000,72600,my_adv_set_item,
    #                                             cleandatapath,trainlabelpath,desc='RawNet')
    # del(model726)
