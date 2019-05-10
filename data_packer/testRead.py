#Use this tester to check the build of your npy files
#The 10'000 first images should be benign (tag=0) the the 10'000 last, malignant (tag=1)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_array = np.load('../data/BenignAndMalignant20000DatasetIMG.npy')
tags_array = np.load('../data/BenignAndMalignant20000DatasetTAG.npy')

isMalignant = 0

fig = plt.figure()

for i in range(15):
    im = Image.fromarray(img_array[i+(9999*isMalignant)])
    print("TAG(", i+(9999*isMalignant), ") = ", tags_array[i+(9999*isMalignant)])
    
    plt.subplot(3, 5, i+1)
    plt.imshow(im)
    
    plt.xticks([])
    plt.yticks([])

plt.show()
    
