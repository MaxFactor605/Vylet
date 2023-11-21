from FE_train import TripletsDataSetRandom
import matplotlib.pyplot as plt
import numpy as np






ds = TripletsDataSetRandom("./dataset_trp_server/dataset_trp", "group_", 1000, 40, "sample_", augmentation=True)

fig, axs = plt.subplots(5, 5)
sample = 19
for i in range(0, 25):
    #group = np.random.randint(0, 1000)
    if i//5 >= 5:
        break
    axs[i//5][i%5].set_title(i)
    axs[i//5][i%5].imshow(ds[(i+1)*40+sample][0])
    



plt.show()

