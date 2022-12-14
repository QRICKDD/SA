import math
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import tqdm

from Tools.RName import flctonpy


class SA_1D_ens3():
    def __init__(self, attackdir, savedir, model, model_few, model_more,
                 input_shape=(1, 1, 64600), input_shape_more=(1, 1, 72600), input_shape_few=(1, 1, 40000),
                 sub_number=5, gw=True,
                 v_num=10, v_range=(-0.01, 0.01),
                 c_norm=1.0, c_norm_more=2.0, c_norm_few=2.0, c_few=0.5, c_more=0.5,
                 eps=0.05, alpha=0.001, steps=500, decay=1.0, is_rawnet=True):
        self.model = model
        self.v_num = v_num
        self.v_range = v_range
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        self.gw = gw
        self.c_norm = c_norm
        self.c_few = c_few
        self.c_more = c_more
        self.c_norm_more = c_norm_more
        self.c_norm_few = c_norm_few

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
            self.signlelength_more = input_shape_more
            self.signlelength_few = input_shape_few
        else:
            self.signlelength = input_shape[-1]
            self.signlelength_more = input_shape_more[-1]
            self.signlelength_few = input_shape_few[-1]

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
            adv_images_few = []
            adv_images_more = []

            adv_images_normals = []
            adv_images_fews = []
            adv_images_mores = []

            # ??????????????????few
            adv_images_few.append(adv_images[:, :self.signlelength_few])
            # ??????????????????more
            adv_images_more.append(
                torch.cat((adv_images, adv_images[:, :self.signlelength_more - self.signlelength]), dim=-1))

            if self.sub_number != 0:
                # ???????????????silder  fews
                step = int((self.signlelength - self.signlelength_few) / self.sub_number)
                for s in range(0, self.signlelength + step - self.signlelength_few, step):
                    adv_images_fews.append(adv_images[:, s:s + self.signlelength_few])

                # ???????????????silder mores
                multi = math.ceil(self.signlelength_more / self.signlelength)
                temp_multi_audio = torch.cat((adv_images, adv_images), dim=-1)
                step = int((self.signlelength * multi - self.signlelength_more) / self.sub_number)
                for s in range(0, self.signlelength * multi - self.signlelength_more, step):
                    adv_images_mores.append(temp_multi_audio[:, s:s + self.signlelength_more])

            # ???????????????
            adv_images_fews = adv_images_fews[1:]
            adv_images_mores = adv_images_mores[1:]

            if self.v_num != 0:
                # ??????normals
                temp_w = adv_images[0].shape
                for v in range(self.v_num):
                    adv_images_normals.append(adv_images +
                                              torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())
                # ??????more
                n_ = len(adv_images_more)
                for n in range(n_):
                    temp_w = adv_images_more[0].shape
                    for v in range(self.v_num):
                        adv_images_more.append(adv_images_more[n] +
                                               torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())
                n_ = len(adv_images_mores)
                for n in range(n_):
                    temp_w = adv_images_mores[0].shape
                    for v in range(self.v_num):
                        adv_images_mores.append(adv_images_mores[n] +
                                                torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())
                # ??????few
                n_ = len(adv_images_few)
                for n in range(n_):
                    temp_w = adv_images_few[0].shape
                    for v in range(self.v_num):
                        adv_images_few.append(adv_images_few[n] +
                                              torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())
                n_ = len(adv_images_fews)
                for n in range(n_):
                    temp_w = adv_images_fews[0].shape
                    for v in range(self.v_num):
                        adv_images_fews.append(adv_images_fews[n] +
                                               torch.Tensor(temp_w).uniform_(self.v_range[0], self.v_range[1]).cuda())

            if self.v_num != 0:
                adv_input_normals = torch.cat(adv_images_normals, dim=0)
                outputs_normals = self.model(adv_input_normals)
                temp_n = outputs_normals.shape[0]  # normals
                normals_labels = target_labels.repeat(temp_n, 1)

            adv_input_few = torch.cat(adv_images_few, dim=0)
            adv_input_more = torch.cat(adv_images_more, dim=0)
            output = self.model(adv_images)
            outputs_few = self.model_few(adv_input_few)
            outputs_more = self.model_more(adv_input_more)
            temp_n = outputs_few.shape[0]  # few
            few_labels = target_labels.repeat(temp_n, 1)
            temp_n = outputs_more.shape[0]  # more
            more_labels = target_labels.repeat(temp_n, 1)

            if self.sub_number != 0:
                adv_input_fews = torch.cat(adv_images_fews, dim=0)
                adv_input_mores = torch.cat(adv_images_mores, dim=0)
                outputs_fews = self.model_few(adv_input_fews)
                outputs_mores = self.model_more(adv_input_mores)
                temp_n = outputs_fews.shape[0]  # fews
                fews_labels = target_labels.repeat(temp_n, 1)
                temp_n = outputs_mores.shape[0]  # mores
                mores_labels = target_labels.repeat(temp_n, 1)

            if self.gw:
                # ?????????????????????
                norm_sc = self.soft(output)[0][1].clone().detach().cpu().item()
                few_sc = self.soft(outputs_few)[0][1].clone().detach().cpu().item()
                more_sc = self.soft(outputs_more)[0][1].clone().detach().cpu().item()
                mymax = self.sub_number * (1 + self.v_num) / 3
                self.c_norm_few = np.clip(1 / (few_sc / (norm_sc + few_sc + more_sc)), 1, mymax)
                self.c_norm_more = np.clip(1 / (more_sc / (norm_sc + few_sc + more_sc)), 1, mymax)
                self.c_norm = np.clip(1 / (norm_sc / (norm_sc + few_sc + more_sc)), 1, mymax)
                # print("c_normal:",self.c_norm,"c_norm_few:",self.c_norm_few,"c_norm_more:",self.c_norm_more)

            cost = -loss(output, target_labels) * self.c_norm
            cost = cost - loss(outputs_more, more_labels) * self.c_norm_more
            cost = cost - loss(outputs_few, few_labels) * self.c_norm_few
            if self.sub_number != 0:
                cost = cost - loss(outputs_mores, mores_labels) * self.c_more
                cost = cost - loss(outputs_fews, fews_labels) * self.c_few

            if self.v_num != 0:
                cost = cost - loss(outputs_normals, normals_labels) * 1

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

            cstep += 1
            # if cstep%10==0:
            #     print(cstep)
            #     print(" model predict", self.soft(output))
            #     print("few model predict", self.soft(outputs_few))
            #     print("more model predict", self.soft(outputs_more))
            #     print("fews model predict", self.soft(outputs_fews))
            #     print("mores model predict", self.soft(outputs_mores))
            #     if self.v_num!=0:
            #         print("normals model predict", self.soft(outputs_normals))

        return adv_images

    def attack(self):
        allnum = 0
        allsuccess = 0
        for item in tqdm.tqdm(self.attackfiles):
            # ?????????  ????????????????????????
            x, sr = sf.read(os.path.join(self.attackdir, item))
            x = x[:self.signlelength]
            x = np.expand_dims(torch.Tensor(x).numpy(), 0)

            # ?????????????????????????????????????????????????????????????????????????????????????????????
            prey = self.model(torch.Tensor(x).cuda()).detach().cpu().numpy()
            # print(prey)
            if np.argmax(prey, axis=1).item() != 0:
                print("??????????????????")
                continue
            allnum += 1

            xadv = self.attack_(torch.Tensor(x), torch.Tensor([[0., 1.]]))
            # ????????????????????????
            prey = self.model(xadv)
            temp_norm_res = np.argmax(prey.detach().cpu().numpy(), axis=1).item()
            # ????????????????????????2
            temp_few_xadv = xadv[:, :self.signlelength_few]
            temp_few_prey = self.model_few(temp_few_xadv)
            temp_few_res = np.argmax(temp_few_prey.detach().cpu().numpy(), axis=1).item()
            # ??????????????????3
            temp_more_xadv = torch.cat((xadv, xadv[:, :self.signlelength_more - self.signlelength]), dim=-1)
            temp_more_prey = self.model_more(temp_more_xadv)
            temp_more_res = np.argmax(temp_more_prey.detach().cpu().numpy(), axis=1).item()

            xadv = xadv.detach().cpu().numpy()
            xadv = xadv.squeeze(0)
            if temp_norm_res != 1 or temp_few_res != 1 or temp_more_res != 1:  # ????????????????????????????????????
                print("????????????")
            else:
                allsuccess += 1
                np.save(os.path.join(self.savedir, flctonpy(item)), xadv)
        # ?????????????????????
        print("???????????????:", allsuccess / allnum)
