import os
import numpy as np

fileNameTAGS = "BenignAndMalignant20000DatasetTAG.npy"
imagesPath = "BenignAndMalignant20000Dataset"

def packImage(fileNameTAGS, imageName):
        
    if os.path.exists(fileNameTAGS) == 0:
        tag_expand = np.expand_dims(imageName[0:1], 0)
        np.save(fileNameTAGS, tag_expand)
    
    else:
        all_tags = np.load(fileNameTAGS)
        tag_expand = np.expand_dims(imageName[0:1], 0)
        tags_cat = np.concatenate((all_tags, tag_expand))
        np.save(fileNameTAGS, tags_cat)
        

i = 1
listOfFiles = os.listdir(imagesPath)
for file in listOfFiles:
    
    print("Packing tag " + str(i) + "/" + str(len(listOfFiles)))
    packImage(fileNameTAGS, file)
    i = i + 1
    
print("Done")
