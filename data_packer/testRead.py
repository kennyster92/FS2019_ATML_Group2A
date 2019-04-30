import numpy as np
from PIL import Image

img_array = np.load('../data/BenignAndMalignant20000DatasetIMG.npy')
tags_array = np.load('../data/BenignAndMalignant20000DatasetTAG.npy')

im = Image.fromarray(img_array[18000])
print("TAG = ", tags_array[18000])

im.show()