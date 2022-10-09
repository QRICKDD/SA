import Tools.trans_tool as tran
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
import yaml
from Config.attackconfig import *

from RawNet646.model import RawNet
#测试到646的迁移性
model_path = Config.RawNet646_model
cleandatapath = Config.TPATH
trainlabelpath = Config.TLABEL
yaml_path = Gconfig.RAW_YAML_CONFIG_PATH


#初始化646模型
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
print("迁移性从4到646:")
for name,advsavepath in zip(Attacks_Name,RawNet4_Advset):
    print(name, end="")
    A = tran.CalClip40000to64600TransAtKSuccess(model,40000,64600,advsavepath,
                                            cleandatapath,trainlabelpath,feature='raw',
                                            desc='RawNet')
    print(A)