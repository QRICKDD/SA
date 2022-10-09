import numpy as np
import matplotlib.pyplot as plt
from Tools.RName import flctonpy,npytoflc

fname=r"C:\Users\djc\Desktop\TTSVC_attack\Adversarial-1D\AavSave\Raw646\PGD\LA_T_1005349.npy"
x646=np.load(fname,allow_pickle=True)

#x726=np.concatenate((x646,x646[:72600-64600]),axis=-1)
x56=x646[:56000]

x726=np.concatenate((x56,x56[:72600-56000]),axis=-1)
x646=np.concatenate((x56,x56[:64600-56000]),axis=-1)
x48=x646[:48000]
x4=x646[:40000]
x=x646
plt.plot(x,c='black')
plt.xlim(0,len(x))
plt.ylim(-1,1)
plt.show()
