import Tools.trans_tool as tran
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
import yaml
from Config.attackconfig import *

from RawNet646.model import RawNet
#测试646到48的迁移性
model_path = Config.RawNet48_model
cleandatapath = Config.TPATH
trainlabelpath = Config.TLABEL
yaml_path = Gconfig.RAW48_YAML_CONFIG_PATH


#初始化48模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
#
model = RawNet(parser1['model'], device)
model = (model).to(device)
model=model.eval()
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))


#循环测试迁移性
print("迁移性从646到48:")
Attacks_Name=[Attacks_Name[-1]]
RawNet646_Advset=[RawNet646_Advset[-1]]
for name,advsavepath in zip(Attacks_Name,RawNet646_Advset):
    print(name, end="")
    A = tran.CalTransAtKSuccess(model,64600,48000,advsavepath,
                                            cleandatapath,trainlabelpath,desc='RawNet')