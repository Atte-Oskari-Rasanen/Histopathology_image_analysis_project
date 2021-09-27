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
def brightness(factor, im): # if factor = 1, then image remains unchanged, if lower, then darkens the image and vice versa
    #image brightness enhancer
    enhancer = ImageEnhance.Brightness(im)
    
    # factor = 1 #gives original image
    # im_output = enhancer.enhance(factor)
    # im_output.save('original-image.png')
    
    factor = 0.5 #darkens the image
    im_output = enhancer.enhance(factor)
    #im_output.save('darkened-image.png')
    im_output = np.asarray(im_output)
    return im_output
    # factor = 1.5 #brightens the image
    # im_output = enhancer.enhance(factor)
    # im_output.save('brightened-image.png')


def remove_horiz_vert_lines(img):
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(img)
    vertical = np.copy(img)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    #for vertical lines:
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    #refine edges
    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical)
    # '''
    # Extract edges and smooth image according to the logic
    # 1. extract edges
    # 2. dilate(edges)
    # 3. src.copyTo(smooth)
    # 4. blur smooth img
    # 5. smooth.copyTo(src, edges)
    # '''
    # # Step 1
    # # Step 2
    # kernel = np.ones((2, 2), np.uint8)
    # edges = cv2.dilate(img, kernel)
    # # Step 3
    # smooth = np.copy(vertical)
    # # Step 4
    # smooth = cv2.blur(smooth, (2, 2))
    # # Step 5
    # (rows, cols) = np.where(edges != 0)
    # vertical[rows, cols] = smooth[rows, cols]
    return vertical
# Load image, grayscale, Otsu's threshold
# h_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15s_hunu_segm.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15sec_col1a1.png'

# h_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30s_hunu_segm.png'
# c_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30sec_col1a1.png'

h_path = '/home/atte/Documents/PD_images/batch6/DAB120/DAB_120s_hunu_segm.png'
c_path = '/home/atte/Documents/PD_images/batch6/DAB120/DAB_120sec_col1a1.png'

# h_path = '/home/inf-54-2020/experimental_cop/All_imgs_segm/segm_batch6/DAB_15s_D_segm.png'
# c_path = '/home/inf-54-2020/experimental_cop/batch6/DAB_15sec_col1a1.png'
# outp = '/home/inf-54-2020/experimental_cop/batch6/batch6_coloc/'
outp = '/home/atte/Documents/PD_images/batch6/output/'
def hunu_ch_import_TH(im_path):
    kernel = np.ones((5,5),np.uint8)

    gray_arr = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    #print(gray_arr.shape)
    thresh = threshold_otsu(gray_arr)
    print(thresh)
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = Image.fromarray(gray_arr)
    #gray = brightness(0.5,gray)
    #cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', gray)
    #blur the image slightly so that some of the smallest particles are removed after thresholding
    im_blur = gray.filter(ImageFilter.GaussianBlur(3)) #If using an image with lots of noise, blur 10 instead of 5!
    im_blur = np.asarray(im_blur)
    (T, thresh_im) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY)
    
    #result = remove_horiz_vert_lines(thresh_im)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    # opening = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel)
    # result = 255 - opening
    # result = cv2.dilate(result,kernel,iterations = 1)

    # distance = ndi.distance_transform_edt(result)
    # coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=result)
    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(coords.T)] = True
    # markers, _ = ndi.label(mask)
    # labels = watershed(-distance, markers, mask=result)

    return thresh_im
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
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(15))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    threshInv = cv2.dilate(threshInv,kernel,iterations = 1)
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

    #print(imagename)

hunu_im = hunu_ch_import_TH(h_path)
cv2.imwrite(outp + 'All_HuNu'+ filename_h, hunu_im)

# out=hunu_im.copy()

col1a1_im = col1a1_ch_import_TH(c_path)
# cv2.imwrite(outp + 'Bin_COL1A1'+ filename_c, col1a1_im)

out=col1a1_im.copy()


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
out[out_mask == 0] = 0


# cv2.imwrite('/home/atte/Documents/PD_images/batch6/coloc_dab.png', canny_output)
#cv2.imwrite(outp + filename_h + 'ou', hunu_im)
cv2.imwrite(outp + 'Coloc_' + filename_h, out)
print('colocalised image created!')
