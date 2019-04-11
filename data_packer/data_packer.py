import os
import numpy as np
import PIL

fileNameIMGS = "BenignAndMalignant20000DatasetIMG.npy"
fileNameTAGS = "BenignAndMalignant20000DatasetTAG.npy"
imagesPath = "BenignAndMalignant20000Dataset"

def packImage(fileNameIMGS, fileNameTAGS, imageName):
    
    image = PIL.Image.open(imagesPath + "/" + imageName)

    if os.path.exists(fileNameIMGS) == 0:
        image_expand = np.expand_dims(image, 0)
        np.save(fileNameIMGS, image_expand)
    
    else:
        all_im = np.load(fileNameIMGS)
        image_expand = np.expand_dims(image, 0)
        img_cat = np.concatenate((all_im, image_expand))
        np.save(fileNameIMGS, img_cat)
            
        
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
    
    print("Packing image " + str(i) + "/" + str(len(listOfFiles)))
    packImage(fileNameIMGS, fileNameTAGS, file)
    i = i + 1
    
print("Done")

