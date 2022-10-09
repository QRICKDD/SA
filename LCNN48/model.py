from torch import nn

from LCNN48.lfcc_model_LCNN import LightCNN_29Layers as LFCCLCNN


class Model_zoo(nn.Module):
    def __init__(self, mt, device=None, parser1=None):
        super(Model_zoo, self).__init__()
        self.mt = mt
        self.device = device
        self.n_m = {
            "lcnn_lfcc": LFCCLCNN(),
        }
        self.model = self.n_m[self.mt]
        self.conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(1000, 2)

    # x input shape = (64600)
    def forward(self, x, Freq_aug=False):
        x = x.unsqueeze(dim=1)  # (1,mmmm,num_filter)
        # 将1d矩阵变成3D矩阵
        x = self.conv(x)  # (1,1,mmm,num_filter)
        x = self.model(x)
        # 将1000分类变成2分类
        x = self.fc(x)

        return x

#
# from ASVBaselineTool.DEMONEED import *
#
#
# def demo():
#     # MFCC 特征
#
#     USE_FEATURE = LFCCfeature
#
#     # 测试lcnn的网络畅通性
#     feaure_net_map = {
#         # MFCCfeature: "lcnn_mfcc",
#         LFCCfeature: "lcnn_lfcc",
#         # CQCCfeature: "lcnn_cqcc",
#         # SPECfeature: "lcnn_spec",
#     }
#
#     for feature in [USE_FEATURE]:
#         net = Model_zoo(feaure_net_map[feature])
#         # 增加batch维度
#         # feature = feature.unsqueeze(dim=0)
#         print(net(feature).shape)
#         print("lcnn 模型 {} 特征输出正常".format(feaure_net_map[feature]))
#
#     # 测试desnet的网络畅通性
#     # for feature in [USE_FEATURE]:
#     #     net = Model_zoo("DensenNet121")
#     #     # 增加batch维度
#     #     # feature = feature.unsqueeze(dim=0)
#     #     print(net(feature).shape)
#     #     print("DensenNet121 模型 mfcc 特征输出正常")
#
#
# if __name__ == "__main__":
#     demo()
