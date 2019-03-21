#Transformations to artificially increase the number of images
#Transformations used: flip, color jitter, rotation

import matplotlib.pyplot as plt
#import torch
import torchvision.transforms as transforms
#from torch.utils.data import Dataset, DataLoader
import PIL
import os
import sys

imgs = []
path = "downloadedFiles/ISIC-images/UDA-1"   


def transformHorizontalFlip(imagePath):
    image = PIL.Image.open(imagePath)
    transform = transforms.RandomHorizontalFlip(p=1.0)
    transformed = transform(image)
    
    transformed.rotate(90).save(path + "/" + os.path.splitext(file)[0] + "_rotated" + os.path.splitext(file)[1])




#def transformHorizontalFlip(imagePath):

 

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
            
            print("Application of the <transformation> to image: " + str(file))



# =============================================================================
# image = PIL.Image.open('cat4.jpg')
# image = image.resize((320, 320))
# 
# plt.imshow(image)
# plt.axis('off');
# 
# #Horizontal flip
# transform = transforms.RandomHorizontalFlip(p=1.0)
# transformed = transform(image)
# plt.imshow(transformed)
# plt.axis('off');
# 
# 
# # =============================================================================
# # #Random crop
# # transform = transforms.RandomCrop((220, 220))
# # f, axes = plt.subplots(2, 3, figsize=(12.8, 9.6))
# # axes = [ax for axs in axes for ax in axs]
# # for i in range(6):
# #     transformed = transform(image)
# #     axes[i].imshow(transformed)
# #     axes[i].axis('off')
# #     
# # 
# #     
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
