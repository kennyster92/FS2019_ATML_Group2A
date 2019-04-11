import os
import numpy as np
import PIL

fileNameIMGS = "BenignAndMalignant20000DatasetIMG.npy"
fileNameTAGS = "BenignAndMalignant20000DatasetTAG.npy"
imagesPath = "BenignAndMalignant20000Dataset"

listOfFiles = os.listdir(imagesPath)

imageArray = []

for file in listOfFiles:
    
    print(imagesPath + "/" + file)
    
    image = PIL.Image.open(imagesPath + "/" + file)
    
    imageArray.append(np.array(image))
 
np.save(fileNameIMGS, imageArray)
print("Done")
