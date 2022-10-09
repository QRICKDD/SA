import matplotlib.pyplot as plt

import numpy as np
import torch
import os

NpPath=r"F:\Adversarial-1D\Paint_Adv2\56_726"


Npscore=os.path.join(NpPath,"score")
Fscore=os.listdir(Npscore)
Fscore=[os.path.join(Npscore,item) for item in Fscore]
Fscore=sorted(Fscore,key=lambda x:int(os.path.basename(x).split(".")[0]))


All_scores=[]
for f in Fscore:
    td=np.load(f,allow_pickle=True)
    if len(td)!=150:
        continue
    All_scores.append(td)

All_scores=np.array(All_scores)
All_scores_self=All_scores[:,:,0]
All_scores_other=All_scores[:,:,1]
All_scores_self_mean=np.mean(All_scores_self,axis=0,keepdims=True)
All_scores_other_mean=np.mean(All_scores_other,axis=0,keepdims=True)
All_scores_self_mean=All_scores_self_mean.reshape(-1,)
All_scores_other_mean=All_scores_other_mean.reshape(-1,)

fontsize=15

linewidth=3

p4,=plt.plot(All_scores_self_mean,c='dodgerblue',linewidth=linewidth,label="ASR-RawNet$_{56}$")
p5,=plt.plot(All_scores_other_mean,c='yellow',linewidth=linewidth,label="ASR-RawNet$_{726}$")
plt.grid(b=True,c='gray',linestyle='--')
plt.xlim(0,150)
plt.ylim(0,1.01)
plt.yticks([0,0.25,0.5,0.75,1],fontsize=fontsize-2)
plt.xticks([0,25,50,75,100,125,150],fontsize=fontsize-2)
plt.ylabel("Attack Success Rate",fontsize=fontsize)
plt.xlabel("Number of iterations",fontsize=fontsize)
plt.legend(handles=[p4,p5], loc ="lower right",fontsize=fontsize)
plt.tight_layout()
plt.savefig(r"F:\Adversarial-1D\Prove\GS-scores-56-726.png",dpi=300)
plt.show()







