from Attacks.FGSM import FGSM
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet646.model import RawNet
import yaml
model_path = Config.RawNet646_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.RawNet646_FGSMAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yaml_path = Gconfig.RAW646_YAML_CONFIG_PATH
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

a = FGSM(attackdir=datasetPath,savedir=AdvsetPath,model=model,input_shape=(1,1,64600))
a.attack(eps=0.06)

