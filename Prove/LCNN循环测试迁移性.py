import torch
import torchaudio

import Config.attackconfig as Config
import Config.globalconfig as Gconfig
import Tools.trans_tool as tran
from LCNN4.model import Model_zoo as LCNN4
from LCNN48.model import Model_zoo as LCNN48
from LCNN56.model import Model_zoo as LCNN56
from LCNN646.model import Model_zoo as LCNN646
from LCNN726.model import Model_zoo as LCNN726

labelPath = Config.TLABEL
datasetPath = Config.TPATH
cleandatapath = Config.TPATH
trainlabelpath = Config.TLABEL

datasetPath = Gconfig.TPATH56000


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


Attacks_Path = [
    r"F:\Adversarial-1D\AavSave\LCNN56\CW-0002-c10",
    r"F:\Adversarial-1D\AavSave\LCNN56\DeepFool-0_1",
    r"F:\Adversarial-1D\AavSave\LCNN56\FGSM",
    r"F:\Adversarial-1D\AavSave\LCNN56\FGSM0015",
    r"F:\Adversarial-1D\AavSave\LCNN56\MIFGSM10-001",
    r"F:\Adversarial-1D\AavSave\LCNN56\MIFGSM20-002",
    r"F:\Adversarial-1D\AavSave\LCNN56\PGD20-002",
    r"F:\Adversarial-1D\AavSave\LCNN56\PGD50-005"
]

# for AdvsetPath in Attacks_Path:
#     print(AdvsetPath, ":", end='')
#     cal_mean_SNR(datasetPath, AdvsetPath)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model646_path = Config.LCNN646_model
model646 = LCNN646("lcnn_lfcc", device)
model646.load_state_dict(torch.load(model646_path, map_location=device))
model646 = NewLCNNModel(model646)
model646 = (model646).to(device)

# 遍历测试迁移性
for my_adv_set_item in Attacks_Path:
    print('开始测试，路径如下')
    print(my_adv_set_item)
    # 测4的迁移性
    model4_path = Config.LCNN4_model
    model4 = LCNN4("lcnn_lfcc", device)
    model4.load_state_dict(torch.load(model4_path, map_location=device))
    model4 = NewLCNNModel(model4)
    model4 = (model4).to(device)
    print('Model loaded : {}'.format(model4_path))
    print("迁移性从56到4:")
    A = tran.CalTransATKTwoModel(model646, model4, 56000, 40000, my_adv_set_item,
                                 cleandatapath, trainlabelpath, desc='LCNN4')
    del (model4)

    # 测48的迁移性
    model48_path = Config.LCNN48_model
    model48 = LCNN48("lcnn_lfcc", device)
    model48.load_state_dict(torch.load(model48_path, map_location=device))
    model48 = NewLCNNModel(model48)
    model48 = (model48).to(device)
    print('Model loaded : {}'.format(model48_path))
    print("迁移性从56到48:")
    A = tran.CalTransATKTwoModel(model646, model48, 56000, 48000, my_adv_set_item,
                                 cleandatapath, trainlabelpath, desc='LCNN48')
    del (model48)

    # 测56的迁移性
    model56_path = Config.LCNN56_model_2
    model56 = LCNN56("lcnn_lfcc", device)
    model56.load_state_dict(torch.load(model56_path, map_location=device))
    model56 = NewLCNNModel(model56)
    model56 = (model56).to(device)
    print('Model loaded : {}'.format(model56_path))
    print("迁移性从56到56*:")
    A = tran.CalTransATKTwoModel(model646, model56, 56000, 56000, my_adv_set_item,
                                 cleandatapath, trainlabelpath, desc='LCNN56')
    del (model56)

    # 测646的迁移性
    print('Model loaded : {}'.format(model646_path))
    print("迁移性从56到646:")
    A = tran.CalTransATKTwoModel(model646, model646, 56000, 64600, my_adv_set_item,
                                 cleandatapath, trainlabelpath, desc='LCNN646')
    del (model646)

    # 测726的迁移性
    model726_path = Config.LCNN726_model
    model726 = LCNN726("lcnn_lfcc", device)
    model726.load_state_dict(torch.load(model726_path, map_location=device))
    model726 = NewLCNNModel(model726)
    model726 = (model726).to(device)
    print('Model loaded : {}'.format(model726_path))
    print("迁移性从56到726:")
    A = tran.CalTransATKTwoModel(model646, model726, 56000, 72600, my_adv_set_item,
                                 cleandatapath, trainlabelpath, desc='LCNN726')
    del (model726)
