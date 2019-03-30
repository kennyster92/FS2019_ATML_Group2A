import os
import sys
import numpy as np
import PIL

def packFiles(path, filename):   
    tmp_data = []
    for file in os.listdir(path):
        image = PIL.Image.open(path + "/" + file)
        
        tmp_data.append(image)
        
    np.savez(filename + ".npy", tmp_data)
    
print("Packaging")

packFiles("downloadedFiles/ISIC-images/UDA-1", "testPackage1")

print("Done")