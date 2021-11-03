
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter
from skimage import measure, filters

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import sys


# # Load image, grayscale, Otsu's threshold
# h_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15s_hunu_segm.png' 
# h_path = '/home/atte/Documents/PD_images/batch6/10ep_Alldat_kaggleDice_S_DAB_Hunu_15sec_512.png'
# h_path = '/home/atte/Documents/PD_images/batch6/DAB15/u2_5ep_bs128_dice_DAB_15sec_512.png'
# h_path ='/home/atte/Desktop/quick/U2_ep3_Alldat_bs128_dice_DAB_15sec_s736.png'
# # h_path = '/home/inf-54-2020/experimental_cop/All_imgs_segm/u2_5ep_bs128_dice_DAB_15sec_512.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15sec_col1a1.png'
# c_path = '/home/atte/Desktop/quick/col1a1_DAB15sec.png'
# c_path = '/home/inf-54-2020/experimental_cop/batch6/DAB_15sec_col1a1.png'
h_path = '/home/atte/Desktop/DAB_30sec_hunu.png736_Segm_ps737_ep5.tif'
c_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30sec_col1a1.png'

# h_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30s_hunu_segm.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30sec_col1a1.png'

# h_path = '/home/atte/Documents/PD_images/batch6/DAB120/DAB_120s_hunu_segm.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB120/DAB_120sec_col1a1.png'

# h_path = '/home/inf-54-2020/experimental_cop/All_imgs_segm/segm_batch6/DAB_15s_D_segm.png'
# c_path = '/home/inf-54-2020/experimental_cop/batch6/DAB_15sec_col1a1.png'
# outp = '/home/inf-54-2020/experimental_cop/batch6/batch6_coloc/'
outp = '/home/atte/Desktop/Testing_coloc/'

#for the images segmented with script U2.py need to invert colours
from PIL import ImageOps
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from scipy.ndimage.filters import gaussian_filter
from skimage import util 

def hunu_ch_import_TH(im_path, radius, sigma):
    img = cv2.imread(im_path,0)
    # img = np.resize(img, (height, width))

    selem = disk(radius)
    im_blur = gaussian_filter(img, sigma=sigma)
    
    print(im_blur.shape)
    local_otsu = rank.otsu(im_blur, selem)
    binary = im_blur >= local_otsu
    print(binary.dtype)
    binary = binary.astype(np.uint8)
    # binary = cv2.bitwise_not(binary) #invert colours so that cells are white and background is black
    # img = np.invert(binary)
    img = util.invert(binary)

    img = Image.fromarray(np.uint8(binary * 255))

    # img.save(outp + 'TEST_'+ filename_h)
    # cv2.imwrite(outp + str(r) +'otsu_local.png', binary)

    # gray_arr = cv2.imread(im_path,0)
    # selem = disk(radius)
    # im_blur = gaussian_filter(gray_arr, sigma=sigma)

    # print(im_blur.shape)
    # local_otsu = rank.otsu(im_blur, selem)
    # binary = im_blur >= local_otsu
    # binary = binary.astype(np.uint8)
    # print(binary.dtype)
    # hunu_im = Image.fromarray(binary)
    # hunu_im.save(outp + 'TEST_'+ filename_h)

#    binary = cv2.bitwise_not(binary) #invert colours so that cells are white and background is black
    # img = Image.fromarray(np.uint8(binary * 255))

    return img
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', threshInv)

def col1a1_ch_import_TH(im_path):
    kernel = np.ones((5,5),np.uint8)

#image = cv2.imread(imagepath)
    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    thresh = threshold_otsu(im_gray)
    
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh - 40
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(im_gray, (7, 7), 0)
    im_gray = Image.fromarray(im_gray)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(20))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    threshInv = cv2.dilate(threshInv,kernel,iterations = 1)
    # np.invert(threshInv)

    return threshInv
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/dab_binary_col1a1.png', threshInv)

# h_path = sys.argv[0]
# c_path = sys.argv[1]
# outp = sys.argv[2]
# print(h_path)
# for imagefile in os.listdir(h_path):  #to go through files in the specific directory
    #print(os.listdir(directory))
# imagepath=directory + "/" + imagefile
# if not imagefile.endswith('.tif') or imagefile.endswith('.jpg'): #exclude files not ending in .tif
#     continue
#print(imagepath)
filename_h = os.path.basename(h_path)
filename_c = os.path.basename(c_path)

col1 = cv2.imread(c_path)
height = col1.shape[0]
width = col1.shape[1]
global height
global width
filename_h = os.path.basename(h_path)
filename_c = os.path.basename(c_path)


hunu_im = hunu_ch_import_TH(h_path, 50, 10)

hunu_im = np.asarray(hunu_im)
#hunu_im = np.resize(hunu_im, (height, width, 3))
#plt.imshow(hunu_im)
hunu_im.save(outp + 'Bin_' + filename_h)
col1a1_im = col1a1_ch_import_TH(c_path)
#cv2.imwrite(outp + 'Bin_COL1A1__'+ filename_c, col1a1_im)

# out=col1a1_im.copy()

# cv2.imwrite('/home/atte/Documents/PD_images/batch6/col1a1.png', col1a1_im)
cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
#_ , contours, _ = cv2.findContours(threshInv,2,1)            
contours = sorted(cnts, key=cv2.contourArea)            #get the largest contour

out_mask = np.zeros_like(hunu_im)

#draw contours of col1a1 image onto the hunu one
# cv2.drawContours(hunu_im, contours, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

#use this when applying mask to the image of nuclei
cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)                                        

#cv2.drawContours(Img, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

out=hunu_im.copy()
out[out_mask == 0] = 255 #makes nuclei white on the black background

# cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
# #_ , contours, _ = cv2.findContours(threshInv,2,1)            
# contours = sorted(cnts, key=cv2.contourArea)            #get the largest contour
# # contours.astype(np.uint8)
# # print(contours)
# out_mask = np.zeros_like(hunu_im)

# #draw contours of col1a1 image onto the hunu one
# # cv2.drawContours(hunu_im, contours, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

# #use this when applying mask to the image of nuclei
# # cv2.drawContours(out_mask, cnts, -1, 255, cv2.FILLED, 1)
# cv2.drawContours(out_mask, cnts, -1, 255)

# #cv2.drawContours(Img, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

# out=hunu_im.copy()
# out =  np.asarray(out)
# # out[out_mask == 0] = 0
# out_mask[out_mask == 0] = out[out_mask == 0]



#####
#if need be blurring and erosion can be applied to remove lines the black lines that overlap the images. Should not happen though
# out = Image.fromarray(out)
# out_blur = out.filter(ImageFilter.GaussianBlur(3)) #If using an image with lots of noise, blur 10 instead of 5!

# # kernel = np.ones((5,5), np.uint8)
# # out_eroded = cv2.erode(out, kernel, iterations=1)
# # cv2.imwrite('/home/atte/Documents/PD_images/batch6/coloc_dab.png', canny_output)
# #cv2.imwrite(outp + filename_h + 'ou', hunu_im)
# from PIL import Image, ImageFilter
# #out_blur = out.filter(ImageFilter.ModeFilter(size=13))
# out.save(outp + 'Blur_Coloc_' + filename_h)
####


cv2.imwrite(outp + 'Blur_Coloc_' + filename_h, out)
print('colocalised image created!')
