from Attacks.SMIFGSM import SMIFGSM
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet646.model import RawNet
import yaml

model_path = Config.RawNet646_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.RawNet646_SMIFGSMAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yaml_path = Gconfig.RAW646_YAML_CONFIG_PATH
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

a = SMIFGSM(attackdir=datasetPath,savedir=AdvsetPath,
            model=model,input_shape=(1,1,64600),
            is_sub_model=False,
            submodel=None,sub_input_shape=(1,1,56000),skip_step=10,
            v_step=10,v_range=(-0.008,0.008),
            eps=0.04, alpha=0.001, steps=30, decay=0.8)
a.attack()
