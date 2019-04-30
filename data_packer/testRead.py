#Use this tester to check the build of your npy files
#The 10'000 first images should be benign (tag=0) the the 10'000 last, malignant (tag=1)

import numpy as np
from PIL import Image

img_array = np.load('BenignAndMalignant20000DatasetIMG.npy')
tags_array = np.load('BenignAndMalignant20000DatasetTAG.npy')


im = Image.fromarray(img_array[18000])
print("TAG = ", tags_array[18000])

im.show()