import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import tqdm

from Tools.RName import flctonpy


class paint_PGD_ens():
    def __init__(self, attackdir, savedir, model, model_few, grad_save_path="",
                 input_shape=(1, 1, 64600), input_shape_few=(1, 1, 40000),
                 sub_number=5,
                 v_num=10, v_range=(-0.01, 0.01),
                 eps=0.05, alpha=0.001, steps=500, is_rawnet=True):
        self.model = model
        self.v_num = v_num
        self.v_range = v_range
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

        self.grad_save_path = grad_save_path
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
        self.model_few = model_few

        if is_rawnet:
            self.model = self.model_bn_to_eval(self.model)
            self.model_few = self.model_bn_to_eval(self.model_few)
        else:
            self.model = self.model.eval()
            self.model_few = self.model_few.eval()
        self.model = self.model.to(self.device)
        self.model_few = self.model_few.to(self.device)

        self.soft = torch.nn.Softmax(dim=-1)

        if type(input_shape) == int:
            self.signlelength = input_shape
            self.signlelength_few = input_shape_few
        else:
            self.signlelength = input_shape[-1]
            self.signlelength_few = input_shape_few[-1]

        if self.signlelength > self.signlelength_few:
            self.grad_save_path_few = os.path.join(self.grad_save_path, "few")
            if os.path.exists(self.grad_save_path_few) == False:
                os.makedirs(self.grad_save_path_few)
        elif self.signlelength < self.signlelength_few:
            self.grad_save_path_more = os.path.join(self.grad_save_path, "more")
            if os.path.exists(self.grad_save_path_more) == False:
                os.makedirs(self.grad_save_path_more)

        self.grad_save_path_score = os.path.join(self.grad_save_path, "score")
        if os.path.exists(self.grad_save_path_score) == False:
            os.makedirs(self.grad_save_path_score)

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

    def cal_grad_sim_(self, grad_a, grad_b):
        score = np.sum(np.sign(grad_a) == np.sign(grad_b)) / grad_b.size
        return score

    def cal_grad_sim(self, grad_a: torch.Tensor, grad_b: torch.Tensor, grad_norm: torch.Tensor, save_grad: list):
        grad_a = grad_a.clone().detach().cpu().numpy()
        grad_b = grad_b.clone().detach().cpu().numpy()
        grad_norm = grad_norm.clone().detach().cpu().numpy()
        grad_a = grad_a.reshape(-1, )
        grad_b = grad_b.reshape(-1, )
        grad_norm = grad_norm.reshape(-1, )
        if self.signlelength > self.signlelength_few:
            grad_a = grad_a[:self.signlelength_few]
            grad_b = grad_b[:self.signlelength_few]
            grad_norm = grad_norm[:self.signlelength_few]
            score = self.cal_grad_sim_(grad_a, grad_b)
            score_norm = self.cal_grad_sim_(grad_a, grad_norm)
            save_grad.append([score, score_norm])
            return save_grad
        else:
            grad_a = grad_a[:self.signlelength_few - self.signlelength]
            grad_c = grad_b[:self.signlelength_few - self.signlelength]  # 前面的
            grad_d = grad_b[self.signlelength:]  # 后面的
            grad_norm = grad_norm[:self.signlelength_few - self.signlelength]  # 前面的
            score_qin = self.cal_grad_sim_(grad_a, grad_c)
            score_hou = self.cal_grad_sim_(grad_a, grad_d)
            score_norm = self.cal_grad_sim_(grad_a, grad_norm)
            save_grad.append([score_qin, score_hou, score_norm])
            return save_grad

    def save_grad_sim(self, data):
        if self.signlelength > self.signlelength_few:
            fnum = len(os.listdir(self.grad_save_path_few)) + 1
            np.save(os.path.join(self.grad_save_path_few, str(fnum) + ".npy"), data)
        else:
            fnum = len(os.listdir(self.grad_save_path_more)) + 1
            np.save(os.path.join(self.grad_save_path_more, str(fnum) + ".npy"), data)

    def save_score(self, data):
        fnum = len(os.listdir(self.grad_save_path_score)) + 1
        np.save(os.path.join(self.grad_save_path_score, str(fnum) + ".npy"), data)

    def cal_score(self, score_self: torch.Tensor, score_other: torch.Tensor, save_score: list):
        score_self = self.soft(score_self)
        score_other = self.soft(score_other)
        score_self = score_self.clone().detach().cpu().numpy()
        score_other = score_other.clone().detach().cpu().numpy()
        score_self = score_self.reshape(-1)
        score_other = score_other.reshape(-1)
        save_score.append([score_self[-1], score_other[-1]])
        return save_score

    def attack_(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        target_labels = labels.clone().detach().to(self.device)
        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        save_grad = []
        save_score = []
        cstep = 0
        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_images_normal = []
            adv_images_fews = []

            temp_norm_adv_images = adv_images.clone().detach()  # 正常长度的样本
            temp_norm_adv_images.requires_grad = True
            # 直接扩展的样本
            temp_adv_images = adv_images.clone().detach()
            if self.signlelength > self.signlelength_few:
                temp_adv_images = temp_adv_images[:, :self.signlelength_few]
            else:
                temp_adv_images = torch.cat(
                    (temp_adv_images, temp_adv_images[:, :self.signlelength_few - self.signlelength]), dim=-1)
            temp_adv_images.requires_grad = True

            if self.sub_number == 0:
                if self.signlelength > self.signlelength_few:
                    adv_images_fews = [adv_images[:, :self.signlelength_few]]
                else:
                    adv_images_fews = [
                        torch.cat((adv_images, adv_images[:, :self.signlelength_few - self.signlelength]), dim=-1)]
            else:
                step = int((self.signlelength - self.signlelength_few) / self.sub_number)
                for s in range(0, self.signlelength + step - self.signlelength_few, step):
                    adv_images_fews.append(adv_images[:, s:s + self.signlelength_few])

            if self.v_num != 0:
                temp_w = adv_images[0].shape
                for v in range(self.v_num):
                    adv_images_normal.append(adv_images +
                                             torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())

                n_ = len(adv_images_fews)
                for n in range(n_):
                    temp_w = adv_images_fews[0].shape
                    for v in range(self.v_num):
                        adv_images_fews.append(adv_images_fews[n] +
                                               torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())

            output = self.model(adv_images)
            outputs_normal = [self.model(item) for item in adv_images_normal]
            outputs_fews = [self.model_few(item) for item in adv_images_fews]

            # 用于记录的不用管
            temp_norm_output = self.model(temp_norm_adv_images)
            temp_norm_output = -loss(temp_norm_output, target_labels)
            temp_norm_grad = torch.autograd.grad(temp_norm_output, temp_norm_adv_images,
                                                 retain_graph=False, create_graph=False)[0]

            temp_output = self.model_few(temp_adv_images)
            temp_cost = -loss(temp_output, target_labels)
            temp_grad = torch.autograd.grad(temp_cost, temp_adv_images,
                                            retain_graph=False, create_graph=False)[0]

            # cost = -loss(outputs, target_labels)
            cost = -loss(output, target_labels)
            for item in outputs_normal:
                cost = cost - loss(item, target_labels)
            for item in outputs_fews:
                cost = cost - loss(item, target_labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            save_score = self.cal_score(score_self=output, score_other=temp_output, save_score=save_score)
            save_grad = self.cal_grad_sim(grad_a=grad, grad_b=temp_grad, grad_norm=temp_norm_grad, save_grad=save_grad)

            # grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            # grad = grad + momentum*self.decay
            # momentum = grad

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

        # 保存至相应文件
        self.save_score(save_score)
        self.save_grad_sim(save_grad)
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
            # 判断是否攻击成功1
            prey = self.model(xadv)
            # 判断是否攻击成功2
            if self.signlelength > self.signlelength_few:
                temp_xadv = xadv[:, :self.signlelength_few]
            else:
                temp_xadv = torch.cat((xadv, xadv[:, :self.signlelength_few - self.signlelength]), dim=-1)

            temp_prey = self.model_few(temp_xadv)
            temp_res = np.argmax(temp_prey.detach().cpu().numpy(), axis=1).item()

            xadv = xadv.detach().cpu().numpy()
            xadv = xadv.squeeze(0)
            if np.argmax(prey.detach().cpu().numpy(), axis=1).item() != 1 or temp_res != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                np.save(os.path.join(self.savedir, flctonpy(item)), xadv)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)
