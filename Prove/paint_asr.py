import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# data = [0.995,1.0,0.064,0.929,0.979,
#         0.418,0.753,0.986,0.991,0.911]

# labels=['CW','Deepfool','FGSM','MI-FGSM','PGD',
#             "PG$D_{ens}$\n56+4",
#             'PG$D_{ens}$\n56+48',
#             'PG$D_{ens}$\n56+646',
#             'PG$D_{ens}$\n56+726',
#             'PG$D_{ens}$\n56+\n726+4']

data = [0.995,0.98,
        1.0,1.0,
        0.064,0.177,0.285,
        0.929,0.983,0.997,
        0.929,0.969,0.979,
        0.418,0.753,0.986,0.991,0.911]

labels=['CW-10','CW-1',
        'Deepfool\n0.0001','Deepfool\n0.001',
        'FGSM\n0.015','FGSM\n0.03','FGSM\n0.06',
        'MI-FGSM\n20','MI-FGSM\n30','MI-FGSM\n50',
        'PGD\n20','PGD\n40','PGD\n50',
        "PG$D_{ens}$\n56+4",
        'PG$D_{ens}$\n56+48',
        'PG$D_{ens}$\n56+646',
        'PG$D_{ens}$\n56+726',
        'PG$D_{ens}$\n56+\n726+4']
fontsize=14
fig = plt.figure(figsize=(10,5))
#fc='dodgerblue'
rects=plt.bar(range(len(data)), data,tick_label=labels,width=0.5)
for item in rects:
    height=item.get_height()
    plt.text(item.get_x()+item.get_width()/2,height,str(height),size=fontsize-2,ha='center',va='bottom')


plt.ylabel("Attack Success Rate",fontsize=fontsize)
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(1.2))
plt.tick_params(labelsize=9)
#plt.xlabel(fontsize=10)
plt.grid(linestyle='-.',axis='y',alpha=0.64)
plt.tight_layout()
#plt.savefig(r"F:\Adversarial-1D\Prove\ASR.png",dpi=600)
plt.show()