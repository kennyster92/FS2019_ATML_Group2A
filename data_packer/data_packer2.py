import os
import numpy as np
#import PIL
from PIL import Image
fileNameIMGS = "BenignAndMalignant20000DatasetIMG.npy"
imagesPath = "BenignAndMalignant20000Dataset"

listOfFiles = os.listdir(imagesPath)

imageArray = []
i = 1

for file in listOfFiles:
    
    print(imagesPath + "/" + file)
    
    # image = PIL.Image.open(imagesPath + "/" + file)
    image = Image.open(imagesPath + "/" + file)

    imageArray.append(np.array(image))
    
    print("Packing image " + str(i) + "/" + str(len(listOfFiles)))
    i = i + 1
 
np.save(fileNameIMGS, imageArray)
print("Done")
