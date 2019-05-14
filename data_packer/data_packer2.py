#Use this packer to create images and tags npy files

import os
import numpy as np
from PIL import Image

fileNameIMGS = "BenignAndMalignant20000DatasetIMG.npy"
fileNameTAGS = "BenignAndMalignant20000DatasetTAG.npy"

imagesPath = "BenignAndMalignant20000Dataset"

listOfFiles = os.listdir(imagesPath)

imageArray = []
tagArray = []
i = 1

for file in listOfFiles:
    print("Packing image " + str(i) + "/" + str(len(listOfFiles)) + " -> " + file)
    
    image = Image.open(imagesPath + "/" + file)
    
    imageArray.append(np.array(image))
    tagArray.append(file[0:1])
    
    i = i + 1
 
print("Writing images npy file...")
np.save(fileNameIMGS, imageArray)

print("Writing tags npy file...")
np.save(fileNameTAGS, tagArray)


print("Done")
