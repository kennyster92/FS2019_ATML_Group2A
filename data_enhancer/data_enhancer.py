#Transformations to artificially increase the number of images
#Transformations used: flip, color jitter, rotation

#import torchvision.transforms as transforms
#import PIL
import os
import sys
import numpy as np
from skimage.transform import downscale_local_mean
from skimage import io
from scipy import misc
import cv2

from PIL import Image, ImageOps

import PIL

imgs = []
path = "downloadedFiles/"

nameOfAFile = "ISIC_0000078.jpg"


# def transformHorizontalFlip(imagePath):
#     print("Application of the flipping to image: " + str(file))
#
#     image = PIL.Image.open(imagePath)
#     transform = transforms.RandomHorizontalFlip(p=1.0)
#     transformed = transform(image)
#
#     transformed.save(path + "/" + os.path.splitext(file)[0] + "_flipped" + os.path.splitext(file)[1])
#
#
# def transformRotation(imagePath):
#     image = PIL.Image.open(imagePath)
#
#     for nbRot in range(1, 4):
#         print("Application of the rotation " + str(nbRot) + " to image: " + str(file))
#
#         transform = transforms.RandomRotation(45)
#         transformed = transform(image)
#
#         transformed.save(path + "/" + os.path.splitext(file)[0] + "_rotated" + str(nbRot) + os.path.splitext(file)[1])
#
# def transformRndCrop(imagePath):
#     image = PIL.Image.open(imagePath)
#
#     for nbCrop in range(1, 4):
#         print("Application of the random crop " + str(nbCrop) + " to image: " + str(file))
#
#         transform = transforms.RandomCrop((220, 220))
#         transformed = transform(image)
#
#         transformed.save(path + "/" + os.path.splitext(file)[0] + "_rndCroped" + str(nbCrop) + os.path.splitext(file)[1])
#
#
# def transformColorJitter(imagePath):
#     print("Application of the color jitter to image: " + str(file))
#
#     image = PIL.Image.open(imagePath)
#
#     transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02)
#     transformed = transform(image)
#
#     transformed.save(path + "/" + os.path.splitext(file)[0] + "_colorJittered" + os.path.splitext(file)[1])
#
#
# def transformBlurringImage(imagePath, blurringFactor):
#     print("Application of the gaussian blurring filter to image: " + str(file))
#     img = cv2.imread(imagePath)
#
#     kernel = np.ones((blurringFactor,blurringFactor),np.float32)/(blurringFactor*blurringFactor)
#     dst = cv2.filter2D(img,-1,kernel)
#
#     cv2.imwrite(imagePath[:-4] + "_blurred.jpg", dst)


def downsampleDataset(imagePath, finalDimX, finalDimY):
    image = Image.open(imagePath)
    
    dimX, dimY = image.size
    
    if dimX != finalDimX or dimY != finalDimY:
        downSamplingFactor = finalDimX/dimX
        
        newDimX = int(dimX/int((1/downSamplingFactor)+1))
        newDimY = int(dimY/int((1/downSamplingFactor)+1))
        
        print("Downsampling image " + str(imagePath) + " to (" + str(newDimX) + ";" + str(newDimY) + ") -> Ratio = %.3f" % downSamplingFactor)
        
        arr = np.array(image)
        image_downscaled = downscale_local_mean(arr, (int((1/downSamplingFactor)+1), int((1/downSamplingFactor)+1), 1))
        
        deltaX = image_downscaled.shape[1] - finalDimX
        deltaY = image_downscaled.shape[0] - finalDimY
        
        print("Black border size [width;height] = (" + str(deltaX) + ";" + str(deltaY) + ")")
        
        if deltaX != 0 or deltaY != 0:
        
            new_size = (finalDimX, finalDimY)
            
            dimX, dimY = image.size
            
            old_size = (image_downscaled.shape[1], image_downscaled.shape[0])
            
            borderX = int((new_size[0]-old_size[0])/2)
            borderY = int((new_size[1]-old_size[1])/2)
            
            border=(borderX, borderY, borderX+1, borderY+1)
            
            PILImg =  Image.fromarray(np.uint8(image_downscaled))            
            image_downscaled = ImageOps.expand(PILImg, border)
            
# =============================================================================
#         if deltaX > 0:
#             print("crop X")
#             
#             
#             
#             new_im = image_downscaled.crop(161, 166, 706, 1050)
# =============================================================================
        
        misc.imsave(imagePath[:-4] + "_downsampled.jpg", image_downscaled)

    else:
        print("Skipping image " + str(imagePath) + "with size " + str(dimX) + ";" + str(dimY))



if os.path.exists(path) == 0 or len(os.listdir(path)) == 0:
    print("Images not found. Put the data set in: downloadedFiles")
    sys.exit()
    
    
valid_images = [".jpg"]
i = 1
listOfFiles = os.listdir(path)
for file in listOfFiles:
    ext = os.path.splitext(file)[1]
    
    if ext.lower() in valid_images:
        if len(file) == len(nameOfAFile):
            print("Transformations applied to image " + str(i) + "/" + str(len(listOfFiles)))

            #transformHorizontalFlip(path + "/" + file)
            #transformRotation(path + "/" + file)
            #transformRndCrop(path + "/" + file)
            #transformColorJitter(path + "/" + file)
            #transformBlurringImage(path + "/" + file, 25)
            downsampleDataset(path + "/" + file, 256, 192)

            i = i + 1

