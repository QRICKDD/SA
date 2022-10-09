import torch
import numpy as np
import soundfile as sf
import torchattacks
import os
import tqdm
from Tools.RName import flctonpy
import torch.nn as nn
import torch.optim as optim

#torchattacks.CW
class CW_Sub_ens_sub():
    def __init__(self, attackdir, savedir, model, model_few, model_ff,
                 input_shape=(1, 1, 64600),input_shape_few=(1,1,40000),input_shape_ff=(1,1,48000),
                 sub_number=5,v_num=0,v_range=(-0.001,0.001),
                 c=1e-1,c_few=1e-1,c_ff=1e-1, kappa=0, steps=500, lr=0.08):
        self.model=model
        self.c = c
        self.v_num=v_num
        self.v_range = v_range
        self.c_few = c_few
        self.c_ff = c_ff
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
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

        self.model_ff=model_ff
        self.model_ff=self.model_bn_to_eval(self.model_ff)
        self.model_ff = self.model_ff.to(self.device)

        self.soft=torch.nn.Softmax(dim=-1)

        if type(input_shape)==int:
            self.signlelength = input_shape
            self.signlelength_few=input_shape_few
            self.signlelength_ff=input_shape_ff
        else:
            self.signlelength = input_shape[-1]
            self.signlelength_few = input_shape_few[-1]
            self.signlelength_ff = input_shape_ff[-1]

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
        labels = labels.clone().detach().to(self.device)

        target_labels = 1

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        w_fews=[]
        w_ff=[]
        if self.sub_number==0:
            w_fews=[w[:,:self.signlelength_few]]
            w_ff = [w[:, :self.signlelength_few]]

        else:
            step=int((self.signlelength-self.signlelength_few)/self.sub_number)
            for s in range(0,self.signlelength+step-self.signlelength_few,step):
                w_fews.append(w[:, s:s+self.signlelength_few])

            step = int((self.signlelength - self.signlelength_ff) / self.sub_number)
            for s in range(0, self.signlelength + step - self.signlelength_ff, step):
                w_ff.append(w[:, s:s + self.signlelength_ff])


        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        #optimizer = optim.Adam([w], lr=self.lr)
        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = [self.tanh_space(w)]

            adv_images_fews =[self.tanh_space(item) for item in w_fews]
            adv_images_ffs = [self.tanh_space(item) for item in w_ff]
            if self.v_num!=0:
                temp_w=adv_images[0].shape
                for v in range(self.v_num):
                    adv_images.append(adv_images[0]+torch.Tensor(temp_w).uniform_(self.v_range[0],self.v_range[1]).cuda())

                n_ = len(w_fews)
                for n in range(n_):
                    temp_w = adv_images_fews[n].shape
                    for v in range(self.v_num):
                        adv_images_fews.append(adv_images_fews[n] +
                                               torch.Tensor(temp_w).uniform_(self.v_range[0],self.v_range[1]).cuda())

                n_ = len(w_ff)
                for n in range(n_):
                    temp_w = adv_images_ffs[n].shape
                    for v in range(self.v_num):
                        adv_images_ffs.append(adv_images_ffs[n] +
                                               torch.Tensor(temp_w).uniform_(self.v_range[0],self.v_range[1]).cuda())

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images[0]),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = [self.model(item) for item in adv_images]
            outputs_fews = [self.model_few(item) for item in adv_images_fews]
            outputs_ffs = [self.model_ff(item) for item in adv_images_ffs]

            f_loss = [self.f(output, target_labels).sum() for output in outputs]
            f_loss_fews = [self.f(item, target_labels).sum() for item in outputs_fews]
            f_loss_ffs = [self.f(item, target_labels).sum() for item in outputs_ffs]
            cost = L2_loss
            for item in f_loss:
                cost=cost+self.c*item
            for item in f_loss_fews:
                cost=cost+self.c_few*item
            for item in f_loss_ffs:
                cost=cost+self.c_ff*item


            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs[0].detach(), 1)
            correct = (pre == labels).float()


            # filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            #mask = mask.view([-1]+[1]*(dim-1))
            #best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
            best_adv_images = adv_images[0].detach()
            best_adv_images=torch.clamp(best_adv_images,-1,1)

            #print(self.model(adv_images))
            #print("===")
            # Early stop when loss does not converge.
            if step % (self.steps//10) == 0:
                for item in outputs:
                    print("normal model predict", self.soft(item))
                for item in outputs_fews:
                    print("few model predict", self.soft(item))
                for item in outputs_ffs:
                    print("ff model predict", self.soft(item))
                print("cost:",cost.item())
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images


    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        return torch.clamp((i-j), min=-self.kappa)

    def tanh_space(self, x):
        return torch.tanh(x)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

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


            xadv = self.attack_(torch.Tensor(x),torch.Tensor([1]))
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

