# Sliding Attack
Code for Clipping and Self-Splicing Resilient Sliding Attack for Synthetic Speech Detection by Jiacheng Deng, Li Dong.

![avatar](https://github.com/QRICKDD/SA/blob/main/pic/SA.png)

+ We make the first steps to study the effect of clipping and self-splicing on adversarial examples. By conducting a comprehensive experiment, we observe that existing adversarial attacks are difficult to maintain threatening after clipping or self-splicing.
+ We define gradient similarity, study the gradients of adversarial examples after clipping and self-splicing, and show that sub-segments with the same sampling value are similar among different models.
+ A new adversarial attacking method termed Sliding Attack (**SA**) is proposed.The proposed method can craft adversarial examples that still can fool black-box models after being clipped or self-spliced.


# Dependencies
The code for our paper runs with Python 3.8 and requires Pytorch of version 1.8.1 or higher. Please pip install the following packages:
* numpy
* soudfile
* torchaudio
* pytorch-cuda

# Running in Docker, MacOS or Ubuntu
We provide as an example the source code to run SA Attack on Rawnets and LCNN trained on ASVspoof2021 LF. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/QRICKDD/SA.git
cd SA
############################################### 
# Carry out SA attack  on provided samples.
python run RawNet56/attack_SA_1D_3.py


# The results path is configured in Config/attackconfig.py and Config/globalconfig.py
```

See `SA/Attacks` and `SA\Config` for details. 
