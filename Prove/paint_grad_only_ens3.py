import matplotlib.pyplot as plt

import numpy as np
import torch
import os

NpPath=r"F:\Adversarial-1D\Paint_Adv2\56_726_4"

Npcfds=os.path.join(NpPath,"gradscore")
Fcfds=os.listdir(Npcfds)
Fcfds=[os.path.join(Npcfds,item) for item in Fcfds]
Fcfds=sorted(Fcfds,key=lambda x:int(os.path.basename(x).split(".")[0]))


All_more=[]
for f in Fcfds:
    td=np.load(f,allow_pickle=True)
    All_more.append(td)
All_more=np.array(All_more)
All_more_A=All_more[:,:,0]
All_more_B=All_more[:,:,1]
All_more_C=All_more[:,:,2]
All_more_F=All_more[:,:,3]
All_more_A_mean=np.mean(All_more_A,axis=0,keepdims=True)
All_more_B_mean=np.mean(All_more_B,axis=0,keepdims=True)
All_more_C_mean=np.mean(All_more_C,axis=0,keepdims=True)
All_more_F_mean=np.mean(All_more_F,axis=0,keepdims=True)
All_more_A_mean=All_more_A_mean.reshape(-1,)
All_more_B_mean=All_more_B_mean.reshape(-1,)
All_more_C_mean=All_more_C_mean.reshape(-1,)
All_more_F_mean=All_more_F_mean.reshape(-1,)

fontsize=15
linewidth=3

p1,=plt.plot(All_more_A_mean,c='black',linestyle='-',linewidth=linewidth,label="Avg.$GS_{A^*}$")
p2,=plt.plot(All_more_B_mean,c='black',linestyle='--',linewidth=linewidth,label="Avg.$GS_{B^*}$")
p3,=plt.plot(All_more_C_mean,c='limegreen',linestyle='-',linewidth=linewidth,label="Avg.$GS_{C^*}$")
p4,=plt.plot(All_more_F_mean,c='bisque',linestyle='-',linewidth=linewidth,label="Avg.$GS_{E^*}$")

plt.grid(b=True,c='gray',linestyle='--')
plt.xlim(0,150)
plt.ylim(0,1.01)
plt.yticks([0,0.25,0.5,0.75,1],fontsize=fontsize-2)
plt.xticks([0,25,50,75,100,125,150],fontsize=fontsize-2)
plt.ylabel("Gradient Similarity",fontsize=fontsize)
plt.xlabel("Number of iterations",fontsize=fontsize)
plt.legend(handles=[p1,p2,p3,p4], loc ="upper right",fontsize=fontsize)
plt.tight_layout()
plt.savefig(r"F:\Adversarial-1D\Prove\GS-grad-Only-56-726-4.png",dpi=300)
plt.show()


