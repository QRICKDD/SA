import os

import numpy as np
import soundfile as sf
import torch

from Tools.RName import flctonpy, npytoflc


# 获取label和文件名对应的字典
def getlabelmeta(labelpath):
    with open(labelpath, "r") as f:
        Labels = f.readlines()
    AFnames = [item.split(" ")[0] for item in Labels]
    ALabels = [item.split(" ")[-1].strip() for item in Labels]
    # label meta
    LabelMeta = {}
    for f, l in zip(AFnames, ALabels):
        if l == "genuine":
            l = 1
        elif l == "fake":
            l = 0
        LabelMeta[f] = l
    return LabelMeta


def feature_select(wav: torch.Tensor):
    wav = wav.reshape(1, -1)
    return wav


# 计算剪切后的攻击迁移性
# 模型, 原始样本的长度，剪切样本的长度  对抗样本保存的路径 clean语音的路径 训练集的标签 特征提取方式
# desc描述了当前攻击的模型以及对抗样本是攻击哪个模型产生的  必填

def handleSourceTraget(source_length, target_length, perturb):
    if source_length > target_length:
        return perturb[:target_length]
    else:
        return torch.cat((perturb, perturb[:int(target_length - source_length)]), dim=0)


def CalTransATKTwoModel(model1, model2, source_length, target_length, advsavepath, cleandatapath, trainlabelpath,
                        desc: str):
    print("测试剪切的迁移性：原始样本长度{},剪切后样本长度{}".format(source_length, target_length))
    # 首先获取labelmeta
    labelmeta = getlabelmeta(trainlabelpath)
    allvalidf = 0
    alltranf = 0
    alladvfn = os.listdir(advsavepath)
    for fn in alladvfn:
        clean, sr = sf.read(os.path.join(cleandatapath, npytoflc(fn)))
        clean = torch.Tensor(clean)
        cleanfea = feature_select(clean)
        cleanfea = cleanfea.cuda()
        model1.eval()
        batch_x = model1(cleanfea)
        init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
        # 如果本来就预测错误直接pass
        if init_pred != labelmeta[npytoflc(fn)]:
            print("原始分冷错误")
            continue
        else:
            pertub = np.load(os.path.join(advsavepath, flctonpy(fn)), allow_pickle=True)
            if max(abs(pertub)) > 1:
                continue
            pertub = torch.Tensor(pertub)
            if source_length > target_length:
                allvalidf += 3
                gap = source_length - target_length
                pertub1 = pertub[:target_length]
                pertub2 = pertub[gap // 2:target_length + gap // 2]
                pertub3 = pertub[gap:]

                pertubfea1 = feature_select(pertub1).cuda()
                pertubfea2 = feature_select(pertub2).cuda()
                pertubfea3 = feature_select(pertub3).cuda()
                model2.eval()

                batch_x = model2(pertubfea1)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1

                batch_x = model2(pertubfea2)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1

                batch_x = model2(pertubfea3)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1
            else:
                allvalidf += 1
                pertub = handleSourceTraget(source_length, target_length, pertub)
                pertubfea = feature_select(pertub)
                pertubfea = pertubfea.cuda()

                model2.eval()
                batch_x = model2(pertubfea)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1
        print(desc)
        print("攻击迁移成功率:", str(alltranf / allvalidf))
        return


def CalTransAtKSuccess(model, source_length, target_length, advsavepath, cleandatapath, trainlabelpath, desc: str):
    print("测试剪切的迁移性：原始样本长度{},剪切后样本长度{}".format(source_length, target_length))
    # 首先获取labelmeta
    labelmeta = getlabelmeta(trainlabelpath)
    allvalidf = 0
    alltranf = 0
    alladvfn = os.listdir(advsavepath)
    for fn in alladvfn:
        clean, sr = sf.read(os.path.join(cleandatapath, npytoflc(fn)))
        clean = torch.Tensor(clean)
        cleanfea = feature_select(clean)
        cleanfea = cleanfea.cuda()
        model.eval()
        batch_x = model(cleanfea)
        init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
        # 如果本来就预测错误直接pass
        if init_pred != labelmeta[npytoflc(fn)]:
            print("原始分冷错误")
            continue
        else:
            pertub = np.load(os.path.join(advsavepath, flctonpy(fn)), allow_pickle=True)
            if max(abs(pertub)) > 1:
                continue
            pertub = torch.Tensor(pertub)
            if source_length > target_length:
                allvalidf += 3
                gap = source_length - target_length
                pertub1 = pertub[:target_length]
                pertub2 = pertub[gap // 2:target_length + gap // 2]
                pertub3 = pertub[gap:]

                pertubfea1 = feature_select(pertub1).cuda()
                pertubfea2 = feature_select(pertub2).cuda()
                pertubfea3 = feature_select(pertub3).cuda()
                model.eval()

                batch_x = model(pertubfea1)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1

                batch_x = model(pertubfea2)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1
                batch_x = model(pertubfea3)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1
            else:
                allvalidf += 1
                pertub = handleSourceTraget(source_length, target_length, pertub)
                pertubfea = feature_select(pertub)
                pertubfea = pertubfea.cuda()

                model.eval()
                batch_x = model(pertubfea)
                init_pred = torch.max(batch_x, dim=1)[1].item()  # 计算最大预测
                if init_pred != labelmeta[npytoflc(fn)]:
                    alltranf += 1
    print(desc)
    print("攻击迁移成功率:", str(alltranf / allvalidf))
    return
