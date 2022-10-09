import torch.nn as nn
import torch
import numpy as np
import soundfile as sf
from art.attacks.evasion import HopSkipJump
from Attacks.AttackFramework import AtkFWork
import os
import tqdm
class HSJA(AtkFWork):
    def __init__(self,attackdir,savedir,model,input_shape=(1,1,64600)):
        super(HSJA, self).__init__(attackdir=attackdir, savedir=savedir, model=model, input_shape=input_shape)
    def attack(self,batch_size=1,max_iter=50,max_eval=10000,init_eval=100,init_size=100,verbose=True):
        allnum=0
        allsuccess=0
        attack = HopSkipJump(classifier=self.classifier,
                             batch_size=batch_size,
                             max_iter=max_iter,
                             max_eval=max_eval,
                             init_eval=init_eval,
                             init_size=init_size,
                             norm=2,verbose=verbose)
        for item in tqdm.tqdm(self.attackfiles):
            # 读数据  切片成需要的长度
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(),0)
            #x=torch.Tensor(x).numpy()

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
            print(prey)
            if np.argmax(prey,axis=1).item() != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                xadv = xadv.squeeze(0)
                sf.write(os.path.join(self.savedir, item), xadv, 16000)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)






