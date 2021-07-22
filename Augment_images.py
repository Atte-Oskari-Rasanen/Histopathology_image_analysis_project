import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A
from tifffile import imread
import imgaug


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


#for file in os.listdir(directory): #open the directory and find the filename, then get the path based on this

image = imread('/home/atte/Documents/images_qupath2/cropped_20x.tif')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

v = visualize(image)

transform = A.HorizontalFlip(p=0.5)
print('a')
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image)
print('b')
transform = A.ShiftScaleRotate(p=0.5)
random.seed(7) 
augmented_image = transform(image=image)['image']
visualize(augmented_image)


transform = A.ShiftScaleRotate(p=0.5)
random.seed(7) 
augmented_image = transform(image=image)['image']
visualize(augmented_image)
print('c')

transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.augmentations.transforms.GaussNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.augmentations.geometric.transforms.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.augmentations.transforms.Sharpen(),
            A.augmentations.transforms.Emboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])
random.seed(42) 
augmented_image = transform(image=image)['image']
visualize(augmented_image)
print('lol')