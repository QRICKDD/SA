import torch
import torchaudio

import Config.attackconfig as Config
from Attacks.CW import CW
from Attacks.DeepFool import DeepFool
from Attacks.FGSM import FGSM
from Attacks.MIFGSM import MIFGSM
from Attacks.PGD import PGD
from LCNN48.model import Model_zoo

model_path = Config.LCNN48_model
labelPath = Config.TLABEL
datasetPath = Config.TPATH
AdvsetPath = Config.LCNN48_FGSMAdvsetPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = Model_zoo("lcnn_lfcc")
model = (model).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))


class NewLCNN48Model(torch.nn.Module):
    def __init__(self, model):
        super(NewLCNN48Model, self).__init__()
        self.LFCC = torchaudio.transforms.LFCC(n_lfcc=60)
        self.model = model

    def forward(self, x: torch.Tensor):
        x_inp = self.LFCC(x)
        delta = torchaudio.functional.compute_deltas(x_inp)
        delta2 = torchaudio.functional.compute_deltas(delta)
        lfccs = torch.concat([x_inp, delta, delta2], dim=1)
        lfccs = torch.transpose(lfccs, 1, 2)
        output = model(lfccs)
        return output


new_model = NewLCNN48Model(model)

# CW  0.998
AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\CW-0002-c10"
a = CW(attackdir=datasetPath, savedir=AdvsetPath, model=new_model, input_shape=(1, 1, 48000),
       c=10, lr=0.002, steps=100, is_raw=False)
a.attack()
del (a)

# DeepFool
AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\DeepFool-eps1e-3-myw1"
a = DeepFool(attackdir=Config.TPATH, savedir=AdvsetPath, model=new_model, input_shape=(1, 1, 48000))
a.attack(eps=0.01, myw=1, max_iter=50, verbose=False)
del (a)

# FGSM
AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\FGSM0015"
a = FGSM(attackdir=datasetPath, savedir=AdvsetPath, model=new_model, input_shape=(1, 1, 48000))
a.attack(eps=0.015)
del (a)

# MIFGSM
AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\MIFGSM10-001"
a = MIFGSM(model=new_model, attackdir=datasetPath, savedir=AdvsetPath, input_shape=(1, 1, 48000),
           eps=0.01, alpha=0.001, steps=10, decay=1, is_raw=False)
a.attack()
del (a)

AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\MIFGSM20-002"
a = MIFGSM(model=new_model, attackdir=datasetPath, savedir=AdvsetPath, input_shape=(1, 1, 48000),
           eps=0.02, alpha=0.001, steps=20, decay=1, is_raw=False)
a.attack()
del (a)

# PGD
AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\PGD20-002"
a = PGD(attackdir=datasetPath, savedir=AdvsetPath, model=new_model, input_shape=(1, 1, 48000))
a.attack(eps=0.05, epstep=0.001, maxiter=50)
del (a)

AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\PGD50-005"
a = PGD(attackdir=datasetPath, savedir=AdvsetPath, model=new_model, input_shape=(1, 1, 48000))
a.attack(eps=0.02, epstep=0.001, maxiter=20)
del (a)
