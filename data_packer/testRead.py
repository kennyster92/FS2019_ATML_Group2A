import numpy as np
from PIL import Image

img_array = np.load('BenignAndMalignant20000DatasetIMG.npy')

im = Image.fromarray(img_array[21])

im.show()