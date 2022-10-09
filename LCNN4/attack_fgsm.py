import torch
import torchaudio

import Config.attackconfig as Config
from Attacks.FGSM import FGSM
from LCNN4.model import Model_zoo

model_path = Config.LCNN4_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.LCNN4_FGSMAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = Model_zoo("lcnn_lfcc")
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))


class NewLCNN4Model(torch.nn.Module):
    def __init__(self, model):
        super(NewLCNN4Model, self).__init__()
        self.LFCC = torchaudio.transforms.LFCC(n_lfcc=60)
        self.model = model

    def forward(self, x: torch.Tensor):
        x_inp = self.LFCC(x)
        delta = torchaudio.functional.compute_deltas(x_inp)
        delta2 = torchaudio.functional.compute_deltas(delta)
        lfccs = torch.concat([x_inp, delta, delta2], dim=1)  # (1，201，180)
        lfccs = torch.transpose(lfccs, 1, 2)
        output = model(lfccs)
        return output


new_model = NewLCNN4Model(model)

a = FGSM(attackdir=datasetPath, savedir=AdvsetPath, model=new_model, input_shape=(1, 1, 40000))
a.attack(eps=0.015)
