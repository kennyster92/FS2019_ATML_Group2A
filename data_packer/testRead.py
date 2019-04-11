import numpy as np
from PIL import Image

#img_array = np.load('BenignAndMalignant20000DatasetIMG.npy')
tag_array = np.load('BenignAndMalignant20000DatasetTAG.npy')

print(tag_array[9999])

#im = Image.fromarray(img_array[100])

#im.show()