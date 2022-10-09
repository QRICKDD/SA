import matplotlib.pyplot as plt

import numpy as np
import torch
import os

NpPath=r"F:\Adversarial-1D\Paint_Adv2\56_4"

Npcfds=os.path.join(NpPath,"few")
Fcfds=os.listdir(Npcfds)
Fcfds=[os.path.join(Npcfds,item) for item in Fcfds]
Fcfds=sorted(Fcfds,key=lambda x:int(os.path.basename(x).split(".")[0]))


All_few=[]
for f in Fcfds:
    td=np.load(f,allow_pickle=True)
    All_few.append(td)
All_few=np.array(All_few)
All_few_norm=All_few[:,:,1]
All_few=All_few[:,:,0]
All_few_mean=np.mean(All_few,axis=0,keepdims=True)
All_few_norm_mean=np.mean(All_few_norm,axis=0,keepdims=True)
All_few_mean=All_few_mean.reshape(-1,)
All_few_norm_mean=All_few_norm_mean.reshape(-1,)


fontsize=15
linewidth=3
p1,=plt.plot(All_few_mean,c='black',linewidth=linewidth,label="Avg.$GS_E$")
p2,=plt.plot(All_few_norm_mean,c='limegreen',linestyle='-',linewidth=linewidth,label="Avg.$GS_D$")

plt.grid(b=True,c='gray',linestyle='--')
plt.xlim(0,150)
plt.ylim(0,0.5)
plt.yticks([0.5,0.6,0.7,0.8,0.9,1],fontsize=fontsize-2)
plt.xticks([0,25,50,75,100,125,150],fontsize=fontsize-2)
plt.ylabel("Gradient Similarity",fontsize=fontsize)
plt.xlabel("Number of iterations",fontsize=fontsize)
plt.legend(handles=[p2,p1], loc ="upper right",fontsize=fontsize)
plt.tight_layout()
plt.savefig(r"F:\Adversarial-1D\Prove\GS-grad-Only-56-4.png",dpi=300)
plt.show()







