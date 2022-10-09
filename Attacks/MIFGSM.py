import torch
import numpy as np
import soundfile as sf
import os
import tqdm
from Attacks.AttackFramework import AtkFWork
from Tools.RName import flctonpy
import torchattacks

class MIFGSM():
    def __init__(self, attackdir, savedir, model,input_shape=(1, 1, 64600),
                 eps=0.05, alpha=0.001, steps=30, decay=1.0,is_raw=True):
        self.attackfiles = os.listdir(attackdir)
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.attackdir = attackdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.model = model
        self.is_raw=is_raw
        if self.is_raw:
            self.model_bn_to_eval()
        else:
            self.model=self.model.eval()
        self.model = self.model.to(self.device)



        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha

        if type(input_shape)==int:
            self.signlelength = input_shape
        else:
            self.signlelength = input_shape[-1]
    def model_bn_to_eval(self):
        self.model.train()
        self.model.first_bn.eval()
        self.model.block0[0].bn2.eval()
        self.model.block1[0].bn1.eval()
        self.model.block1[0].bn2.eval()
        self.model.block2[0].bn1.eval()
        self.model.block2[0].bn2.eval()
        self.model.block3[0].bn1.eval()
        self.model.block3[0].bn2.eval()
        self.model.block4[0].bn1.eval()
        self.model.block4[0].bn2.eval()
        self.model.block5[0].bn1.eval()
        self.model.block5[0].bn2.eval()
        self.model.bn_before_gru.eval()

    def attack_(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        target_labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = torch.nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)


            cost = -loss(outputs, target_labels)


            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images

    def attack(self):
        allnum = 0
        allsuccess = 0
        for item in tqdm.tqdm(self.attackfiles):
            # 读数据  切片成需要的长度
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(),0)

            # 判断本来是非被预测错误：输入全部是假样本，预测为真则本来就错误
            prey = self.model(torch.Tensor(x).cuda()).detach().cpu().numpy()
            if np.argmax(prey,axis=1).item() != 0:
                print("原始分类错误")
                continue
            allnum += 1

            # PGD攻击
            xadv = self.attack_(torch.Tensor(x),torch.Tensor([[0.0,1.0]]))

            # 判断是否攻击成功
            prey = self.model(xadv)
            #print(prey)

            xadv = xadv.detach().cpu().numpy()
            xadv = xadv.squeeze(0)

            if np.argmax(prey.detach().cpu().numpy(),axis=1).item() != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                np.save(os.path.join(self.savedir, flctonpy(item)),xadv)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)






