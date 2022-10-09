import torch
import torchaudio

import Config.attackconfig as Config
from Attacks.SA_1D_3 import SA_1D_ens3
from LCNN4.model import Model_zoo as LCNN4
from LCNN56.model import Model_zoo as LCNN56
from LCNN726.model import Model_zoo as LCNN726
from Prove.paint_pgd_ens import paint_PGD_ens
from Prove.paint_pgd_ens3 import paint_PGD_ens3

labelPath = Config.TLABEL
datasetPath = Config.TPATH
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NewLCNNModel(torch.nn.Module):
    def __init__(self, model):
        super(NewLCNNModel, self).__init__()
        self.LFCC = torchaudio.transforms.LFCC(n_lfcc=60)
        self.model = model

    def forward(self, x: torch.Tensor):
        x_inp = self.LFCC(x)
        delta = torchaudio.functional.compute_deltas(x_inp)
        delta2 = torchaudio.functional.compute_deltas(delta)
        lfccs = torch.concat([x_inp, delta, delta2], dim=1)  # (120,324)
        lfccs = torch.transpose(lfccs, 1, 2)
        output = self.model(lfccs)
        return output


model4_path = Config.LCNN4_model
model4 = LCNN4("lcnn_lfcc", device)
model4.load_state_dict(torch.load(model4_path, map_location=device))
model4 = NewLCNNModel(model4)
model4 = (model4).to(device)

model56_path = Config.LCNN56_model
model56 = LCNN56("lcnn_lfcc", device)
model56.load_state_dict(torch.load(model56_path, map_location=device))
model56 = NewLCNNModel(model56)
model56 = (model56).to(device)

model726_path = Config.LCNN726_model
model726 = LCNN726("lcnn_lfcc", device)
model726.load_state_dict(torch.load(model726_path, map_location=device))
model726 = NewLCNNModel(model726)
model726 = (model726).to(device)


def PGD_double(model1, model2, shape1, shape2):
    AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\TempPaint_" + str(shape1) + "_" + str(shape2)
    Npsave = r"F:\Adversarial-1D\AavSave\LCNN56\PGD_" + str(shape1) + "_" + str(shape2)
    print(AdvsetPath)
    print(Npsave)
    a = paint_PGD_ens(
        attackdir=datasetPath, savedir=AdvsetPath, grad_save_path=Npsave,
        model=model1, model_few=model2,
        sub_number=0, v_num=0, v_range=(-0.01, 0.01),
        input_shape=(1, 1, shape1), input_shape_few=(1, 1, shape2),
        eps=0.05, alpha=0.001, steps=150, is_rawnet=False)
    a.attack()
    del (a)


def PGD_triple():
    AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\TempPaint_4_56_726"
    Npsave = r"F:\Adversarial-1D\AavSave\LCNN56\PGD_4_56_726"
    print(AdvsetPath)
    print(Npsave)
    a = paint_PGD_ens3(
        attackdir=datasetPath, savedir=AdvsetPath, grad_save_path=Npsave,
        model=model56, model_few=model4, model_more=model726,
        input_shape=(1, 1, 56000), input_shape_more=(1, 1, 72600),
        input_shape_few=(1, 1, 40000),
        eps=0.05, alpha=0.001, steps=50, is_rawnet=False)
    a.attack()
    del (a)


def SA_triple(sub_number, v_num):
    AdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\SA-1D\56-726-4-sn{}-v{}".format(
        sub_number, v_num)
    print(AdvsetPath)
    a = SA_1D_ens3(
        attackdir=datasetPath, savedir=AdvsetPath,
        model=model56, model_few=model4, model_more=model726,
        input_shape=(1, 1, 56000), input_shape_few=(1, 1, 40000), input_shape_more=(1, 1, 72600),
        c_norm=1, c_norm_more=1.0, c_norm_few=1.0, c_more=1, c_few=1, gw=False,
        sub_number=sub_number,
        v_num=v_num, v_range=(-0.01, 0.01),
        eps=0.05, alpha=0.001, steps=50, decay=0.1, is_rawnet=False)
    a.attack()
    del (a)


if __name__ == "__main__":
    # PGD_double(model56, model4, 56000, 40000)
    # PGD_double(model56, model726, 56000, 72600)
    # PGD_triple()
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
