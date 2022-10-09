# -- coding: utf-8 --
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from RawNet48.data_utils import genSpoof_list, Dataset_ASVspoof2019_train
from RawNet48.model import RawNet
import Config.globalconfig as Gconfig

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    # 正样本判错率
    num_error_true = 0
    # 负样本判正率
    num_error_fake = 0
    model.eval()
    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (loss) functions
    weight = torch.FloatTensor([0.1,0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in train_loader:

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def mytrain():
    """
    配置参数
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_database_path = Gconfig.TPATH48000
    dev_database_path = Gconfig.EPATH48000
    train_label_path = Gconfig.TLABEL48000
    dev_label_path = Gconfig.ELABEL48000
    batch_size = Gconfig.RAW48_MS_BATCH_SIZE
    num_epochs = Gconfig.RAW48_MS_NUM_EPOCHS
    lr = 0.0003
    weight_decay = 0.00001
    loss = "weighted_CCE"
    # 记录模型保存路径
    model_save_path = Gconfig.RAW48_MS_SAVE_PATH
    model_tag = 'model_{}_{}_1024_lr0.0003_norm'.format(num_epochs, batch_size) # 需要调整的代码 用于保存到当前路径models下模型文件夹的名称
    abs_root = os.getcwd()
    model_save_path = os.path.join(abs_root, model_save_path, model_tag)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # 模型内部结构配置
    yaml_path = Gconfig.RAW48_YAML_CONFIG_PATH
    with open(yaml_path, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml,Loader=yaml.FullLoader)
    # 是否加载已有模型
    is_load_saved_model = False
    haven_saved_model_path = ""
    """
    配置结束
    """

    # 加载模型、配置优化器
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = (model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 是否加载已有模型继续训练
    if is_load_saved_model == True:
        model.load_state_dict(torch.load(haven_saved_model_path, map_location=device))
        print('Model loaded : {}'.format(haven_saved_model_path))

    # 加载训练数据和验证数据
    data_train_label, data_train_file = genSpoof_list(train_label_path, is_eval=False)
    print('no. of training trials', len(data_train_file))
    train_set = Dataset_ASVspoof2019_train(list_IDs=data_train_file,
                                           labels=data_train_label,
                                           base_dir=train_database_path)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    del train_set, data_train_label

    data_dev_label, data_dev_file = genSpoof_list(dev_label_path, is_eval=False)
    print('no. of validation trials', len(data_dev_file))
    dev_set = Dataset_ASVspoof2019_train(list_IDs=data_dev_file,
                                         labels=data_dev_label,
                                         base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    del dev_set, data_dev_label

    # 训练模型
    best_acc = 90
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, lr, optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        print('train_accuracy', train_accuracy, epoch)
        print('valid_accuracy', valid_accuracy, epoch)
        print('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))

        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
        best_acc = max(valid_accuracy, best_acc)
        torch.save(model.state_dict(), os.path.join(model_save_path,
                                                    'epoch_{}_{}_{}.pth'.format(epoch, train_accuracy // 0.1 / 10,
                                                                                valid_accuracy // 0.1 / 10)))


if __name__=="__main__":
    mytrain()