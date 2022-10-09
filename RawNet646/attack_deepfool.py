from Attacks.DeepFool import DeepFool
import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import torch
from RawNet646.model import RawNet
import yaml

model_path = Config.RawNet646_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.RawNet646_DeepfoolAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yaml_path = Gconfig.RAW_YAML_CONFIG_PATH
with open(yaml_path, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
model = RawNet(parser1['model'], device)
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

#import torchaudio
# class NewModel(torch.nn.Module):
#     def __init__(self,model):
#         super(NewModel, self).__init__()
#         self.SPEC=torchaudio.transforms.MFCC()
#         self.model=model
#     def forward(self,x:torch.Tensor):
#         x.requires_grad=True
#         x=self.SPEC(x)
#         output=model(x)
#         return output
# new_model=NewModel(model)

a = DeepFool(attackdir=Config.TPATH,savedir=AdvsetPath,model=model,
             input_shape=(1,1,64600))
a.attack(eps=1e-4,myw=0.8,max_iter=50,verbose=False)
