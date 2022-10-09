import os

import numpy as np
import soundfile as sf
import torch
import tqdm
from art.attacks.evasion import ProjectedGradientDescentPyTorch

from Attacks.AttackFramework import AtkFWork
from Tools.RName import flctonpy


class PGD(AtkFWork):
    def __init__(self, attackdir, savedir, model, input_shape=(1, 1, 64600)):
        super(PGD, self).__init__(attackdir=attackdir, savedir=savedir, model=model, input_shape=input_shape)

    def attack(self, eps=0.3, epstep=0.01, maxiter=50):
        allnum = 0
        allsuccess = 0
        attack = ProjectedGradientDescentPyTorch(estimator=self.classifier, eps=eps, targeted=True,
                                                 eps_step=epstep, max_iter=maxiter, batch_size=1, verbose=False)
        for item in tqdm.tqdm(self.attackfiles):
            # 读数据  切片成需要的长度
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(), 0)

            # 判断本来是非被预测错误：输入全部是假样本，预测为真则本来就错误
            prey = self.classifier.predict(x)
            if np.argmax(prey, axis=1).item() != 0:
                print("原始分类错误")
                continue
            allnum += 1

            # PGD攻击
            xadv = attack.generate(x=x, y=np.array([1.0]))

            # 判断是否攻击成功
            prey = self.classifier.predict(xadv)
            # print(prey)
            if np.argmax(prey, axis=1).item() != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                xadv = xadv.squeeze(0)
                np.save(os.path.join(self.savedir, flctonpy(item)), xadv)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)
