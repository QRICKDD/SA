import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import tqdm

from Tools.RName import flctonpy


class paint_PGD_ens3():
    def __init__(self, attackdir, savedir, model, model_more, model_few, grad_save_path="",
                 input_shape=(1, 1, 64600), input_shape_more=(1, 1, 72600), input_shape_few=(1, 1, 40000),
                 eps=0.05, alpha=0.001, steps=500, is_rawnet=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

        self.grad_save_path = grad_save_path
        self._supported_mode = ['default', 'targeted']

        self.attackfiles = os.listdir(attackdir)
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.attackdir = attackdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape

        self.model = model
        self.model_few = model_few
        self.model_more = model_more

        if is_rawnet:
            self.model = self.model_bn_to_eval(self.model)
            self.model_few = self.model_bn_to_eval(self.model_few)
            self.model_more = self.model_bn_to_eval(self.model_more)
        else:
            self.model = self.model.eval()
            self.model_few = self.model_few.eval()
            self.model_more = self.model_more.eval()

        self.model = self.model.to(self.device)
        self.model_few = self.model_few.to(self.device)
        self.model_more = self.model_more.to(self.device)

        self.soft = torch.nn.Softmax(dim=-1)

        if type(input_shape) == int:
            self.signlelength = input_shape
            self.signlelength_few = input_shape_few
            self.signlelength_more = input_shape_more[-1]
        else:
            self.signlelength = input_shape[-1]
            self.signlelength_few = input_shape_few[-1]
            self.signlelength_more = input_shape_more[-1]

        self.grad_save_path_few = os.path.join(self.grad_save_path, "gradscore")
        if os.path.exists(self.grad_save_path_few) == False:
            os.makedirs(self.grad_save_path_few)

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

    # grad_a是合成的 grad_b的726  grad_c是56 grad_f是4
    def cal_grad_sim(self, grad_ens: torch.Tensor, grad_b: torch.Tensor, grad_c: torch.Tensor, grad_f: torch.Tensor,
                     save_grad: list):
        grad_ens = grad_ens.clone().detach().cpu().numpy()
        grad_b = grad_b.clone().detach().cpu().numpy()
        grad_c = grad_c.clone().detach().cpu().numpy()
        grad_f = grad_f.clone().detach().cpu().numpy()
        grad_ens = grad_ens.reshape(-1, )
        grad_b = grad_b.reshape(-1, )
        grad_c = grad_c.reshape(-1, )
        grad_f = grad_f.reshape(-1, )

        # 计算相似度C*和F*
        grad_ens = grad_ens[:self.signlelength_more - self.signlelength]

        grad_C = grad_c[:self.signlelength_more - self.signlelength]
        grad_F = grad_f[:self.signlelength_more - self.signlelength]
        score_C = self.cal_grad_sim_(grad_ens, grad_C)
        score_F = self.cal_grad_sim_(grad_ens, grad_F)

        # 计算相似度A* B*
        grad_A = grad_b[:self.signlelength_more - self.signlelength]  # 前面的
        grad_B = grad_b[self.signlelength:]  # 后面的
        score_A = self.cal_grad_sim_(grad_ens, grad_A)
        score_B = self.cal_grad_sim_(grad_ens, grad_B)
        save_grad.append([score_A, score_B, score_C, score_F])
        return save_grad

    def save_grad_sim(self, data):
        fnum = len(os.listdir(self.grad_save_path_few)) + 1
        np.save(os.path.join(self.grad_save_path_few, str(fnum) + ".npy"), data)

    def save_score(self, data):
        fnum = len(os.listdir(self.grad_save_path_score)) + 1
        np.save(os.path.join(self.grad_save_path_score, str(fnum) + ".npy"), data)

    # 输入为56 726 4
    def cal_score(self, score_self: torch.Tensor, score_other: torch.Tensor, score_4: torch.Tensor, save_score: list):
        score_self = self.soft(score_self)
        score_other = self.soft(score_other)
        score_4 = self.soft(score_4)
        score_self = score_self.clone().detach().cpu().numpy()
        score_other = score_other.clone().detach().cpu().numpy()
        score_4 = score_4.clone().detach().cpu().numpy()
        score_self = score_self.reshape(-1)
        score_other = score_other.reshape(-1)
        score_4 = score_4.reshape(-1)
        save_score.append([score_self[-1], score_other[-1], score_4[-1]])
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
            adv_images_fews = []
            adv_images_more = []

            # --------------------------------------------------------------------------------------------------------------#
            # 正常长度的样本
            temp_norm_adv_images = adv_images.clone().detach()
            temp_norm_adv_images.requires_grad = True
            # 直接扩展的样本
            temp_more_adv_images = adv_images.clone().detach()
            temp_more_adv_images = torch.cat(
                (temp_more_adv_images, temp_more_adv_images[:, :self.signlelength_more - self.signlelength]), dim=-1)
            temp_more_adv_images.requires_grad = True
            # 短的样本
            temp_few_adv_images = adv_images.clone().detach()
            temp_few_adv_images = temp_few_adv_images[:, :self.signlelength_few]
            temp_few_adv_images.requires_grad = True

            # --------------------------------------------------------------------------------------------------------------#
            # 添加到循环其实不循环中
            adv_images_fews = [adv_images[:, :self.signlelength_few]]  # 56剪切到4
            adv_images_mores = [torch.cat((adv_images, adv_images[:, :self.signlelength_more - self.signlelength]),
                                          dim=-1)]  # 56自拼接到726
            # 计算融合损失
            output = self.model(adv_images)  # 计算正常损失
            outputs_mores = [self.model_more(item) for item in adv_images_mores]  # 计算726损失
            outputs_fews = [self.model_few(item) for item in adv_images_fews]  # 计算4损失
            cost = -loss(output, target_labels)
            for item in outputs_mores:
                cost = cost - loss(item, target_labels)
            for item in outputs_fews:
                cost = cost - loss(item, target_labels)
            # 计算融合梯度
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # --------------------------------------------------------------------------------------------------------------#
            # 记录输入56的梯度
            temp_norm_output = self.model(temp_norm_adv_images)
            temp_norm_cost = -loss(temp_norm_output, target_labels)
            temp_norm_grad = torch.autograd.grad(temp_norm_cost, temp_norm_adv_images,
                                                 retain_graph=False, create_graph=False)[0]

            # 记录输入4的梯度
            temp_few_output = self.model_few(temp_few_adv_images)
            temp_few_cost = -loss(temp_few_output, target_labels)
            temp_few_grad = torch.autograd.grad(temp_few_cost, temp_few_adv_images,
                                                retain_graph=False, create_graph=False)[0]

            # 记录输入726的梯度
            temp_more_output = self.model_more(temp_more_adv_images)
            temp_more_cost = -loss(temp_more_output, target_labels)
            temp_more_grad = torch.autograd.grad(temp_more_cost, temp_more_adv_images,
                                                 retain_graph=False, create_graph=False)[0]
            #
            # print("56:",self.soft(temp_norm_output))
            # print("726:",self.soft(temp_more_output))
            # print("4:",self.soft(temp_few_output))

            save_score = self.cal_score(score_self=output, score_other=temp_more_output,
                                        score_4=temp_few_output, save_score=save_score)
            save_grad = self.cal_grad_sim(grad_ens=grad, grad_b=temp_more_grad, grad_c=temp_norm_grad,
                                          grad_f=temp_few_grad, save_grad=save_grad)

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

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
            temp_norm_res = np.argmax(prey.detach().cpu().numpy(), axis=1).item()
            # 判断是否攻击成功2
            temp_few_xadv = xadv[:, :self.signlelength_few]
            temp_few_prey = self.model_few(temp_few_xadv)
            temp_few_res = np.argmax(temp_few_prey.detach().cpu().numpy(), axis=1).item()
            # 判断成果3
            temp_more_xadv = torch.cat((xadv, xadv[:, :self.signlelength_more - self.signlelength]), dim=-1)
            temp_more_prey = self.model_more(temp_more_xadv)
            temp_more_res = np.argmax(temp_more_prey.detach().cpu().numpy(), axis=1).item()

            xadv = xadv.detach().cpu().numpy()
            xadv = xadv.squeeze(0)
            if temp_norm_res != 1 or temp_few_res != 1 or temp_more_res != 1:  # 如果预测不是真则攻击失败
                print("攻击失败")
            else:
                allsuccess += 1
                np.save(os.path.join(self.savedir, flctonpy(item)), xadv)
        # 输出攻击成功率
        print("攻击成功率:", allsuccess / allnum)
