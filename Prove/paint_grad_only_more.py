import matplotlib.pyplot as plt

import numpy as np
import torch
import os

NpPath=r"F:\Adversarial-1D\Paint_Adv2\56_726"

Npcfds=os.path.join(NpPath,"more")
Fcfds=os.listdir(Npcfds)
Fcfds=[os.path.join(Npcfds,item) for item in Fcfds]
Fcfds=sorted(Fcfds,key=lambda x:int(os.path.basename(x).split(".")[0]))



All_more=[]
for f in Fcfds:
    td=np.load(f,allow_pickle=True)
    All_more.append(td)
All_more=np.array(All_more)
All_more_qian=All_more[:,:,0]
All_more_hou=All_more[:,:,1]
All_more_norm=All_more[:,:,-1]
All_more_qian_mean=np.mean(All_more_qian,axis=0,keepdims=True)
All_more_hou_mean=np.mean(All_more_hou,axis=0,keepdims=True)
All_more_norm_mean=np.mean(All_more_norm,axis=0,keepdims=True)
All_more_qian_mean=All_more_qian_mean.reshape(-1,)
All_more_hou_mean=All_more_hou_mean.reshape(-1,)
All_more_norm_mean=All_more_norm_mean.reshape(-1,)



fontsize=15

linewidth=3
p1,=plt.plot(All_more_qian_mean,c='black',linestyle='-',linewidth=linewidth,label="Avg.$GS_A$")
p2,=plt.plot(All_more_hou_mean,c='black',linestyle='--',linewidth=linewidth,label="Avg.$GS_B$")
p3,=plt.plot(All_more_norm_mean,c='limegreen',linestyle='-',linewidth=linewidth,label="Avg.$GS_C$")

plt.grid(b=True,c='gray',linestyle='--')
plt.xlim(0,150)
plt.ylim(0.5,1.0)
plt.yticks([0.5,0.6,0.7,0.8,0.9,1],fontsize=fontsize-2)
plt.xticks([0,25,50,75,100,125,150],fontsize=fontsize-2)
plt.ylabel("Gradient Similarity",fontsize=fontsize)
plt.xlabel("Number of iterations",fontsize=fontsize)
plt.legend(handles=[p1,p2,p3], loc ="upper right",fontsize=fontsize)
plt.tight_layout()
plt.savefig(r"F:\Adversarial-1D\Prove\GS-grad-Only-56-726.png",dpi=300)
plt.show()
