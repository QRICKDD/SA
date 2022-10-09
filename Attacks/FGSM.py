import torch
import numpy as np
import soundfile as sf
from art.attacks.evasion import FastGradientMethod
from Attacks.AttackFramework import AtkFWork
from Tools.RName import flctonpy
import os
import tqdm
class FGSM(AtkFWork):
    def __init__(self,attackdir,savedir,model,input_shape=(1,1,64600)):
        super(FGSM,self).__init__(attackdir=attackdir, savedir=savedir, model=model, input_shape=input_shape)
    def attack(self,eps=0.08):
        allnum=0
        allsuccess=0
        attack = FastGradientMethod(estimator=self.classifier, eps=eps,batch_size=1,targeted=True)
        for item in tqdm.tqdm(self.attackfiles):
            # 读数据  切片成需要的长度
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(),0)

            # 判断本来是非被预测错误：输入全部是假样本，预测为真则本来就错误
            prey = self.classifier.predict(x)
            if np.argmax(prey,axis=1).item() != 0:
                print("原始分类错误")
                continue
            allnum += 1

            # PGD攻击
            xadv = attack.generate(x=x,y=np.array([1.0]))

            # 判断是否攻击成功
            prey = self.classifier.predict(xadv)
            if np.argmax(prey,axis=1).item() != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                xadv = xadv.squeeze(0)
                np.save(os.path.join(self.savedir, flctonpy(item)),xadv)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)






