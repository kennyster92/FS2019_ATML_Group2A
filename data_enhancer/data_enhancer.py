#Transformations to artificially increase the number of images
#Transformations used: flip, color jitter, rotation

import torchvision.transforms as transforms
import PIL
import os
import sys
import numpy as np

imgs = []
path = "downloadedFiles/ISIC-images/UDA-1"   


def transformHorizontalFlip(imagePath):
    print("Application of the flipping to image: " + str(file))
    
    image = PIL.Image.open(imagePath)
    transform = transforms.RandomHorizontalFlip(p=1.0)
    transformed = transform(image)
    
    transformed.save(path + "/" + os.path.splitext(file)[0] + "_flipped" + os.path.splitext(file)[1])


def transformRotation(imagePath):
    image = PIL.Image.open(imagePath)
    
    for nbRot in range(1, 4):
        print("Application of the rotation " + str(nbRot) + " to image: " + str(file))
        
        transform = transforms.RandomRotation(45)
        transformed = transform(image)
        
        transformed.save(path + "/" + os.path.splitext(file)[0] + "_rotated" + str(nbRot) + os.path.splitext(file)[1])

def transformRndCrop(imagePath): 
    image = PIL.Image.open(imagePath)

    for nbCrop in range(1, 4):
        print("Application of the random crop " + str(nbCrop) + " to image: " + str(file))
        
        transform = transforms.RandomCrop((220, 220))
        transformed = transform(image)
    
        transformed.save(path + "/" + os.path.splitext(file)[0] + "_rndCroped" + str(nbCrop) + os.path.splitext(file)[1])


def transformColorJitter(imagePath):
    print("Application of the color jitter to image: " + str(file))
    
    image = PIL.Image.open(imagePath)
    
    transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02)
    transformed = transform(image)

    transformed.save(path + "/" + os.path.splitext(file)[0] + "_colorJittered" + os.path.splitext(file)[1])        


# Check if data set is in the correct folder
if os.path.exists(path) == 0 or len(os.listdir(path)) == 0:
    print("Images not found. Put the data set in: downloadedFiles/ISIC-images/UDA-1 folder")
    sys.exit()
    
    
    
valid_images = [".jpg"]
for file in os.listdir(path):
    ext = os.path.splitext(file)[1]
    
    if ext.lower() in valid_images:
        if len(file) == 16:
            transformHorizontalFlip(path + "/" + file)            
            transformRotation(path + "/" + file)
            #transformRndCrop(path + "/" + file)
            transformColorJitter(path + "/" + file)



# # #Random crop 2
# # transform = transforms.FiveCrop((220, 220))
# # f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# # axes = [ax for axs in axes for ax in axs]
# # transformed = transform(image)
# # for i in range(5):
# #     axes[i].imshow(transformed[i])
# #     axes[i].axis('off')
# # =============================================================================
#     
#     
# #Color jitter
# transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02)
# 
# f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# axes = [ax for axs in axes for ax in axs]
# for i in range(6):
#     transformed = transform(image)
#     axes[i].imshow(transformed)
#     axes[i].axis('off')
#     
#     
# # =============================================================================
# # #Random resized crop
# # transform = transforms.RandomResizedCrop(220, scale=(0.2, 1.0), ratio=(0.75, 1.333))
# # 
# # f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# # axes = [ax for axs in axes for ax in axs]
# # for i in range(6):
# #     transformed = transform(image)
# #     axes[i].imshow(transformed)
# #     axes[i].axis('off')
# # =============================================================================
#     
#     
# #Random rotation
# transform = transforms.RandomRotation(45)
# 
# f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# axes = [ax for axs in axes for ax in axs]
# for i in range(6):
#     transformed = transform(image)
#     axes[i].imshow(transformed)
#     axes[i].axis('off')  
#     
#     
# #Random affine transform
# transform = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.3), shear=10)
# 
# f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# axes = [ax for axs in axes for ax in axs]
# for i in range(6):
#     transformed = transform(image)
#     axes[i].imshow(transformed)
#     axes[i].axis('off')
#     
# #Composition of transforms
# transform = transforms.Compose([
#                     transforms.RandomCrop((220, 220)),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ColorJitter(0.2, 0.6)
#             ])
# 
# f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# axes = [ax for axs in axes for ax in axs]
# for i in range(6):
#     transformed = transform(image)
#     axes[i].imshow(transformed)
#     axes[i].axis('off')
# =============================================================================
