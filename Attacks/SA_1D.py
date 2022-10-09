import math
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import tqdm

from Tools.RName import flctonpy


class SA_1D_ens2():
    def __init__(self, attackdir, savedir, model, model_other,
                 input_shape=(1, 1, 64600), input_shape_other=(1, 1, 40000),
                 sub_number=5,
                 v_num=10, v_range=(-0.01, 0.01),
                 eps=0.05, alpha=0.001, steps=500, decay=1.0, is_rawnet=True):
        self.model = model
        self.v_num = v_num
        self.v_range = v_range
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        self._supported_mode = ['default', 'targeted']
        self.sub_number = sub_number

        self.attackfiles = os.listdir(attackdir)
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.attackdir = attackdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape

        self.model = model
        self.model_other = model_other

        if is_rawnet:
            self.model = self.model_bn_to_eval(self.model)
            self.model_other = self.model_bn_to_eval(self.model_other)
        else:
            self.model = self.model.eval()
            self.model_other = self.model_other.eval()

        self.model = self.model.to(self.device)
        self.model_other = self.model_other.to(self.device)

        self.soft = torch.nn.Softmax(dim=-1)

        if type(input_shape) == int:
            self.signlelength = input_shape
            self.signlelength_other = input_shape_other
        else:
            self.signlelength = input_shape[-1]
            self.signlelength_other = input_shape_other[-1]

    def model_bn_to_eval(self, model):
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

        cstep = 0
        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_images_normals = []
            adv_images_others = []

            if self.sub_number == 0:
                # 攻击小的模型
                if self.signlelength > self.signlelength_other:
                    adv_images_others = [adv_images[:, :self.signlelength]]
                # 攻击大的模型
                else:
                    adv_images_others = [
                        torch.cat((adv_images, adv_images[:, :self.signlelength_other - self.signlelength]), dim=-1)]
            else:
                # 攻击小模型
                if self.signlelength > self.signlelength_other:
                    step = int((self.signlelength - self.signlelength_other) / self.sub_number)
                    for s in range(0, self.signlelength + step - self.signlelength_other, step):
                        adv_images_others.append(adv_images[:, s:s + self.signlelength_other])
                # 攻击大模型
                else:
                    # 计算扩增倍数 向上取整
                    multi = math.ceil(self.signlelength_other / self.signlelength)
                    temp_multi_audio = torch.cat((adv_images, adv_images), dim=-1)
                    step = int((self.signlelength * multi - self.signlelength_other) / self.sub_number)
                    for s in range(0, self.signlelength * multi - self.signlelength_other, step):
                        adv_images_others.append(temp_multi_audio[:, s:s + self.signlelength_other])

            if self.v_num != 0:
                temp_w = adv_images[0].shape
                for v in range(self.v_num):
                    adv_images_normals.append(adv_images +
                                              torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())

                n_ = len(adv_images_others)
                for n in range(n_):
                    temp_w = adv_images_others[0].shape
                    for v in range(self.v_num):
                        adv_images_others.append(adv_images_others[n] +
                                                 torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())

            output = self.model(adv_images)
            outputs_normal = [self.model(item) for item in adv_images_normals]
            outputs_fews = [self.model_other(item) for item in adv_images_others]

            # cost = -loss(outputs, target_labels)
            cost = -loss(output, target_labels)
            for item in outputs_normal:
                cost = cost - loss(item, target_labels)
            for item in outputs_fews:
                cost = cost - loss(item, target_labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

            # if cstep % (self.steps//10) == 0:
            #     print("normal model predict", self.soft(output))
            #     for item in outputs_normal:
            #         print("normal model predict", self.soft(item))
            #     for item in outputs_fews:
            #         print("few model predict", self.soft(item))

            cstep += 1

        return adv_images

    def attack(self):
        allnum = 0
        allsuccess = 0
        for item in tqdm.tqdm(self.attackfiles):
            # 读数据  切片成需要的长度
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(), 0)

            # 判断本来是非被预测错误：输入全部是假样本，预测为真则本来就错误
            prey = self.model(torch.Tensor(x).cuda()).detach().cpu().numpy()
            # print(prey)
            if np.argmax(prey, axis=1).item() != 0:
                print("原始分类错误")
                continue
            allnum += 1

            xadv = self.attack_(torch.Tensor(x), torch.Tensor([[0., 1.]]))
            # 判断是否攻击成功
            prey = self.model(xadv)
            temp_norm_res = np.argmax(prey.detach().cpu().numpy(), axis=1).item()
            # 判断是否攻击成功2
            if self.signlelength > self.signlelength_other:
                temp_few_xadv = xadv[:, :self.signlelength_other]
                temp_few_prey = self.model_other(temp_few_xadv)
                temp_res = np.argmax(temp_few_prey.detach().cpu().numpy(), axis=1).item()
            else:
                temp_more_xadv = torch.cat((xadv, xadv[:, :self.signlelength_other - self.signlelength]), dim=-1)
                temp_more_prey = self.model_other(temp_more_xadv)
                temp_res = np.argmax(temp_more_prey.detach().cpu().numpy(), axis=1).item()

            xadv = xadv.detach().cpu().numpy()
            xadv = xadv.squeeze(0)
            if temp_norm_res != 1 or temp_res != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                np.save(os.path.join(self.savedir, flctonpy(item)), xadv)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)
