import os

import torch
from torch.utils.data import DataLoader

import Config.globalconfig as Gconfig
from LCNN726.data_utils import genSpoof_list, Dataset_ASVspoof2019_train
# from model import RawGAT_ST  # In main model script we used our best RawGAT-ST-mul model. To use other models you need to call revelant model scripts from RawGAT_models folder
from LCNN726.model import Model_zoo
from RawNet726.train_model import evaluate_accuracy, train_epoch


def train():
    batch_size = Gconfig.LCNN726_MS_BATCH_SIZE
    num_epochs = Gconfig.LCNN726_MS_NUM_EPOCHS
    lr = 0.0001
    weight_decay = 0.0001
    loss = "WCE"
    eval_output = None
    eval = False
    is_eval = False
    # 存放模型权重的路径
    modelth = Gconfig.LCNN726_MS_SAVE_PATH
    # 训练集
    train_path = Gconfig.TPATH72600
    train_label = Gconfig.TLABEL72600
    eval_path = Gconfig.EPATH72600
    eval_label = Gconfig.ELABEL72600

    # define model saving path
    model_tag = '{}_{}_{}_{}'.format(loss, num_epochs, batch_size, lr)
    model_save_path = os.path.join(modelth, model_tag)
    if os.path.exists(model_save_path) is False:
        os.makedirs(model_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    model = Model_zoo("lcnn_lfcc")
    model = (model).to(device)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training Dataloader
    data_train_label, data_train_file = genSpoof_list(train_label, is_eval=False)
    print('no. of training trials', len(data_train_file))
    train_set = Dataset_ASVspoof2019_train(list_IDs=data_train_file,
                                           labels=data_train_label,
                                           base_dir=train_path)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    del train_set, data_train_label

    data_dev_label, data_dev_file = genSpoof_list(eval_label, is_eval=False)
    print('no. of validation trials', len(data_dev_file))
    dev_set = Dataset_ASVspoof2019_train(list_IDs=data_dev_file,
                                         labels=data_dev_label,
                                         base_dir=eval_path)
    dev_loader = DataLoader(dev_set, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    del dev_set, data_dev_label

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


if __name__ == "__main__":
    train()
