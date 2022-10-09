import torch
import numpy as np
import soundfile as sf
import os
import tqdm
from Tools.RName import flctonpy
import torch.nn as nn


class PGD_ens_2():
    def __init__(self, attackdir, savedir, model, model_few,
                 input_shape=(1, 1, 64600),input_shape_few=(1,1,40000),
                 sub_number=5,
                 v_num=10,v_range=(-0.01,0.01),
                 eps=0.05, alpha=0.001,steps=500,decay=1.0):
        self.model=model
        self.v_num=v_num
        self.v_range=v_range
        self.eps=eps
        self.alpha=alpha
        self.steps = steps
        self.decay=decay

        self._supported_mode = ['default', 'targeted']
        self.sub_number=sub_number

        self.attackfiles = os.listdir(attackdir)
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.attackdir = attackdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape=input_shape
        self.model = model
        self.model=self.model_bn_to_eval(self.model)
        self.model = self.model.to(self.device)

        self.model_few=model_few
        self.model_few=self.model_bn_to_eval(self.model_few)
        self.model_few = self.model_few.to(self.device)


        self.soft=torch.nn.Softmax(dim=-1)

        if type(input_shape)==int:
            self.signlelength = input_shape
            self.signlelength_few=input_shape_few
        else:
            self.signlelength = input_shape[-1]
            self.signlelength_few = input_shape_few[-1]

    def model_bn_to_eval(self,model):
        model.train()
        model.first_bn.eval()
        model.block0[0].bn2.eval()
        model.block1[0].bn1.eval()
        model.block1[0].bn2.eval()
        model.block2[0].bn1.eval()
        model.block2[0].bn2.eval()
        model.block3[0].bn1.eval()
        model.block3[0].bn2.eval()
        model.block4[0].bn1.eval()
        model.block4[0].bn2.eval()
        model.block5[0].bn1.eval()
        model.block5[0].bn2.eval()
        model.bn_before_gru.eval()
        return model


    def attack_(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        target_labels = labels.clone().detach().to(self.device)
        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()


        cstep=0
        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_images_normal=[]
            adv_images_fews =[]

            if self.sub_number==0:
                adv_images_fews=[adv_images[:,:self.signlelength]]
            else:
                step=int((self.signlelength-self.signlelength_few)/self.sub_number)
                for s in range(0,self.signlelength+step-self.signlelength_few,step):
                    adv_images_fews.append(adv_images[:, s:s+self.signlelength_few])

            if self.v_num!=0:
                temp_w = adv_images[0].shape
                for v in range(self.v_num):
                    adv_images_normal.append(adv_images+
                                      torch.Tensor(temp_w).uniform_(self.v_range[0],self.v_range[1]).cuda())

                n_ = len(adv_images_fews)
                for n in range(n_):
                    temp_w = adv_images_fews[0].shape
                    for v in range(self.v_num):
                        adv_images_fews.append(adv_images_fews[n] +
                                               torch.Tensor(temp_w).uniform_(self.v_range[0],self.v_range[1]).cuda())

            output = self.model(adv_images)
            outputs_normal = [self.model(item) for item in adv_images_normal]
            outputs_fews = [self.model_few(item) for item in adv_images_fews]

            #cost = -loss(outputs, target_labels)
            cost=-loss(output, target_labels)
            for item in outputs_normal:
                cost=cost-loss(item, target_labels)
            for item in outputs_fews:
                cost=cost-loss(item, target_labels)


            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

            # if cstep % (self.steps//10) == 0:
            #     print("normal model predict", self.soft(output))
            #     for item in outputs_normal:
            #         print("normal model predict", self.soft(item))
            #     for item in outputs_fews:
            #         print("few model predict", self.soft(item))

            cstep+=1

        return adv_images


    def attack(self):
        allnum=0
        allsuccess=0
        for item in tqdm.tqdm(self.attackfiles):
            # 读数据  切片成需要的长度
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(),0)

            # 判断本来是非被预测错误：输入全部是假样本，预测为真则本来就错误
            prey = self.model(torch.Tensor(x).cuda()).detach().cpu().numpy()
            #print(prey)
            if np.argmax(prey,axis=1).item() != 0:
                print("原始分类错误")
                continue
            allnum += 1


            xadv = self.attack_(torch.Tensor(x),torch.Tensor([[0.,1.]]))
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

