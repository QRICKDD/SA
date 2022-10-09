import os
import torch
from art.estimators.classification import PyTorchClassifier

class AtkFWork():
    def __init__(self, attackdir, savedir, model, input_shape=(1, 1, 64600)):
        if type(input_shape)==int:
            self.signlelength = input_shape
        else:
            self.signlelength = input_shape[-1]
        self.attackfiles = os.listdir(attackdir)
        self.savedir = savedir
        if os.path.exists(self.savedir)==False:
            os.makedirs(self.savedir)
        self.attackdir = attackdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_classes = 2
        self.loss = torch.nn.CrossEntropyLoss()
        self.clip_values = (-1.0, 1.0)
        model = model.eval()
        model = model.to(self.device)
        self.classifier = PyTorchClassifier(
            model=model,
            clip_values=self.clip_values,
            loss=self.loss,
            input_shape=input_shape,
            nb_classes=self.nb_classes,
        )


