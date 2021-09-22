import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter

#takes in the image channel with the nuclei and the channel with the cytosolic marker. Processes them (gaussian blurring and thresholding), finds the contours
#of the cytosolic marker channel and apply it to the nuclei channel, generating an image with only nuclei colocalised with the marker.

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

# Load image, grayscale, Otsu's threshold
h_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15s_hunu_segm.png'
c_path = '/home/atte/Documents/PD_images/batch6/DAB15/DAB_15sec_col1a1.png'

h_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30s_hunu_segm.png'
c_path = '/home/atte/Documents/PD_images/batch6/DAB30/DAB_30sec_col1a1.png'

def hunu_ch_import_TH(im_path):
    gray_arr = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    print(gray_arr.shape)
    thresh = threshold_otsu(gray_arr)
    print(thresh)
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = Image.fromarray(gray_arr)
    #gray = brightness(0.5,gray)
    #cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', gray)
    #blur the image slightly so that some of the smallest particles are removed after thresholding
    im_blur = gray.filter(ImageFilter.GaussianBlur(5))
    im_blur = np.asarray(im_blur)
    (T, thresh_im) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    opening = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel)
    print(opening)
    result = 255 - opening
    return result
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/t.png', threshInv)

def col1a1_ch_import_TH(im_path):

#image = cv2.imread(imagepath)
    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    thresh = threshold_otsu(im_gray)
    
    #add extra on top of otsu's thresholded value as otsu at times includes background noise
    thresh = thresh - 40
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(im_gray, (7, 7), 0)
    im_gray = Image.fromarray(im_gray)
    im_blur = im_gray.filter(ImageFilter.GaussianBlur(10))
    im_blur = np.asarray(im_blur)
    
    (T, threshInv) = cv2.threshold(im_blur, thresh, 255,
    	cv2.THRESH_BINARY_INV)
    return threshInv
#cv2.imwrite('/home/atte/Documents/PD_images/batch6/dab_binary_col1a1.png', threshInv)

hunu_im = hunu_ch_import_TH(h_path)
cv2.imwrite('/home/atte/Documents/PD_images/batch6/hunu_morph.png', hunu_im)

col1a1_im = col1a1_ch_import_TH(c_path)
cv2.imwrite('/home/atte/Documents/PD_images/batch6/col1a1.png', col1a1_im)


cnts, _ = cv2.findContours(col1a1_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
#_ , contours, _ = cv2.findContours(threshInv,2,1)            
contours = sorted(cnts, key=cv2.contourArea)            #get the largest contour

out_mask = np.zeros_like(hunu_im)

#draw contours of col1a1 image onto the hunu one
cv2.drawContours(hunu_im, contours, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

#use this when applying mask to the image of nuclei
cv2.drawContours(out_mask, [contours[-1]], -1, 255, cv2.FILLED, 1)                                        

#cv2.drawContours(Img, cnts, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line

out=hunu_im.copy()
out[out_mask == 0] = 0


# cv2.imwrite('/home/atte/Documents/PD_images/batch6/coloc_dab.png', canny_output)
cv2.imwrite('/home/atte/Documents/PD_images/batch6/coloc_dab30_v2_col_outlines.png', hunu_im)
cv2.imwrite('/home/atte/Documents/PD_images/batch6/coloc_dab30_v2.png', out)
print('colocalised image created!')

