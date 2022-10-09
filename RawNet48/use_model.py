import torch
from RawNet646.model import RawNet
import Config.globalconfig as Gconfig
import yaml

def getSavedModel():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #已有的模型路径
    MODEL_SAVE_PATH=Gconfig.RAW48_MS_CHOICE_PATH
    # 模型内部结构配置
    yaml_path = Gconfig.RAW48_YAML_CONFIG_PATH
    with open(yaml_path, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    #
    model = RawNet(parser1['model'], device)
    model = (model).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print('Model loaded : {}'.format(MODEL_SAVE_PATH))
    return model

def demo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=getSavedModel()
    x=torch.randn(1,48000)
    if device=="cuda":
        x=x.cuda()
    batch_x=model(x)
    print(batch_x)
    print(batch_x.shape)
    print("转换成概率")
    print("变成（-1，2）")
    batch_x=batch_x.reshape(-1)
    som=torch.nn.Softmax()
    y=som(batch_x)
    print(y)
    print(y.shape)


if __name__=="__main__":
    demo()