#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 22:34:22 2021

@author: atte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:01:36 2021

@author: atte
"""
import numpy as np

import cv2
from skimage import measure, filters
import os 
import pandas as pd
import sys
import glob



def calculations_hunu(img):
    stats_list = []
    binary = np.asarray(img).astype(np.uint8)
    binary = cv2.bitwise_not(binary)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    contour_area = cv2.contourArea(cnts[0])
    # filename_h = os.path.basename(im_path)
    # stats_list.append(filename_h)
    contour_areas = []
    for cell in range(len(cnts)):
        contour_area = cv2.contourArea(cnts[cell])
        contour_areas.append(contour_area)
    
    total_area_nuclei = sum(contour_areas)
    print('Total area taken up by the nuclei: '+ str(sum(contour_areas)))
    stats_list.append(total_area_nuclei)
    cell_count_coloc=binary.max() #count the number of objects
    stats_list.append(cell_count_coloc)
    labels_h = measure.label(hunu_cells) #create a binary format of the numpy array (object or background)
    cell_count_h=labels_h.max() #count the number of objects
    stats_list.append(cell_count_h)
    total_a = binary.shape[0] * binary.shape[1] 
    # contour_area = cv2.contourArea(contours[0])
    stats_list.append(total_a)
    return(stats_list)

def calculations_col1a1(img):
    stats_list = []
    binary = np.asarray(img).astype(np.uint8)
    binary = cv2.bitwise_not(binary)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get the contours of the col1a1 
    contour_area = cv2.contourArea(cnts[0])
    # filename_c = os.path.basename(im_path)
    # stats_list.append(filename_c)

    contour_areas = []
    for cell in range(len(cnts)):
        contour_area = cv2.contourArea(cnts[cell])
        contour_areas.append(contour_area)

    total_area_col1 = sum(contour_areas)
    print('Total area taken up by the nuclei: '+ str(sum(contour_areas)))
    stats_list.append(total_area_col1)
    filename_h = os.path.basename(im_path)
    total_a = binary.shape[0] * binary.shape[1] 
    # contour_area = cv2.contourArea(contours[0])
    stats_list.append(total_a)
    return(stats_list)
# ws_coloc_path = ['/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Coloc_DAB_15s_hunu_segm.png',
#                 '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Coloc_DAB_30s_hunu_segm.png',
#                 '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Coloc_DAB_120s_hunu_segm.png'] 
# ws_hunu_path = ['/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/DAB_15s_hunu_segm.pngoutlines.png',
#                 '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/DAB_30s_hunu_segm.pngoutlines.png',
#                 '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/DAB_120s_hunu_segm.pngoutlines.png'] 


# hunu_cells = sys.argv[1]
# coloc_cells = sys.argv[2]
# hunu_cells = sys.argv[1]
# coloc_cells = sys.argv[2]

# ws_hunu_path = glob.glob(hunu_cells)
# ws_coloc_path = glob.glob(ws_coloc_path)
# print(ws_hunu_path)
#iterate over the colocalised images directory and save the locations into a list from which 
#the images are retrieved one by one:


# dictionary = dict(zip(ws_hunu_path, ws_coloc_path))

# '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/'


segm_TH_dirs = []
all_ims_paths = []

main_dir = './Deconvolved_ims2'

#get all image paths:
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

#get all images that match the pattern
# matches_list = []
for f in all_ims_paths:
    #print(f)
    im_name = f.split('/')[-1]
    # print(im_name)
    n = 18
    im_id = a = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
    im_id = im_id[0]
    match_hunu_col1 = list(filter(lambda x: im_id in x, all_ims_paths))
    # matches_list.append(match_hunu_col1)
    for file_path in match_hunu_col1:
        filename = os.path.basename(file_path)
        print(filename)

# h_c_list = []
# hunu_nuclei_areas = []
# col1a1_areas = []
# full_im_area = []
# coloc_list = []
# unit_list = []
# filenames = []

hunu_stats_dict = {} #the value part contains: [filename, nuclei A, nuclei N, total_A]
hunu_coloc_stats_dict = {} 

from skimage.filters import threshold_otsu, rank

#gather the information and calculate the stats (no of nuclei and total area they take up)
# for the hunu image with all nuclei and for the colocalised image with the nuclei
# colocalised with col1a1 as well as col1a1 area.

#The first set of loops focuses on extracting the info from hunu images (pure hunu and the
# colocalised ones). The second loop goes through the col1a1 ones. They should be matched to the
#colocalised image with the corresponding id and this should be added into the column.
import scandir
for root, subdirectories, files in scandir.walk(main_dir):
    for subdir in subdirectories:
        if 'Coloc' in subdir:
            for file in files:
                if 'WS' in file:
                    hunu_coloc_stats_list = []
                    filename = file.split('.')[0]
                    print(file)
                    im_path = root + '/' + subdir + file
                    im_gray = cv2.imread(im_path,0)
                    
                    Stats = calculations_hunu(im_gray)
                    hunu_coloc_stats_dict[filename]=Stats
                    
        if 'hunu' in subdir:
            for file in files:
                if 'WS' in file:
                    hunu_stats_list = []
                    file = file.split('.')[0]
                    print(file)
                    # filenames.append(file)

                    im_path = root + '/' + subdir + file
                    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                    Stats = calculations_hunu(im_gray)
                    hunu_stats_dict[filename]=Stats


#Now I have a dictionary with the key being the imagename (pure hunu image or coloc one) and values
#containing the relevant info (Nuclei Area, Nuclei count and total Area of the image-this last one is a sanity check)

#Iterate over the COL1A1 images, count the area of each one's stain, then match to the dict key of the hunu_coloc dict

#get all im paths so that you can get the matching colocalised hunu with col1a1 stains


#you may need to include a constant calculated for each image-pair, taken e.g. from the col1a1 image
#since sometimes the images may not be exactly same size or this could be fixed earlier as well.
all_ims_paths = []
for (dirpath, dirnames, filenames) in os.walk(main_dir):
    all_ims_paths += [os.path.join(dirpath, file) for file in filenames]

col1a1_dict = {}
for root, subdirectories, files in scandir.walk(main_dir):
    for subdir in subdirectories:
        if 'col1a1' in subdir:
            for file in files:
                if 'TH' in file:
                    filename_c = file.split('.')[0]
                    filenames.append(filename_c)
                    im_path = root + '/' + subdir + filename
                    im_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                    stats_col1 = calculations_col1a1(im_gray)
                    col1a1_dict[filename_c] = stats_col1



#get the matches of coloc_hunu, hunu and col1a1 names. Then retrieve the appropriate values like the area from the value part 
#of the dict and do the calculations with the corresponding col1a1 value.

# 22_340886577_15_736_hunu.png
# 22_340886577_15_736_col1a1.png
def getList(dict):
    return dict.keys()

a = {1:[11,12,13], 2:[21,22,23], 3:[31,32,33]}
col1a1_names = getList(col1a1_dict)
df.columns = ["Total # of HuNu cells", "# of Hunu cells colocalised with COL1A1", "N(HUNU)+N(COL1A1+) / N(HUNU+)", "A(HUNU+)A(COL1A1)+ / A(HUNU+)" "A(COL1A1+)/A(HUNU+)", "A(COL1a1)/N(HUNU+)"]
# Nuclei Area, Nuclei count


final_info = {}
#find matching hunu, coloc_hunu and col1a1 ones, get the calculations, save into dict as list value, the key will be the id info of the animal
for key_hunu in hunu_stats_dict.keys():
    for key_hunu_coloc in hunu_coloc_stats_dict.keys():
        for key_col1 in col1a1_dict.keys():
            if key_hunu_coloc[:18] == key_col1[:18] and key_hunu[:18]==key_col1[:18]:
                
                stats = []
                values_hunu_coloc = hunu_coloc_stats_dict[key_hunu_coloc]
                values_col1 = col1a1_dict[key_col1]
                values_hunu = hunu_stats_dict[key_hunu]
                
                #I-symbol used as division
                #Area of hunu-col1a1+ cells / Area of all hunu cells
                Ahunucol1_Acol1_I_Ahunu = values_hunu_coloc[0] / values_hunu[0]
                stats.append(Ahunucol1_Acol1_I_Ahunu)
                #Area of col1a1 / Area of all hunu cells
                Acol1_I_Ahunu = values_col1[0] / values_hunu[0]
                stats.append(Acol1_I_Ahunu)

                #All hunu cells
                Total_hunu_cells = values_hunu[1]
                stats.append(Total_hunu_cells)

                #Hunu cells colocalised with col1a1:
                hunu_coloc = values_hunu_coloc[1]
                stats.append(hunu_coloc)

                #coloc hunu cells / all hunu
                N_hunu_coloc_I_total_hunu = values_hunu_coloc[1] / values_hunu[1]
                stats.append(N_hunu_coloc_I_total_hunu)

                #Area of col1a1 divided by number of hunu cells:
                A_col1_I_total_hunu = values_col1[0] /  values_hunu[1]
                stats.append(A_col1_I_total_hunu)

filenames = final_info.items()
All_stats = list(filenames)

df = pd.DataFrame(All_stats)
df.columns = ["A(HUNU+COL1A1+) / A(HUNU+)", "A(COL1A1+)/A(HUNU+)", "N(HUNU+)", "N(HUNU+COL1A1+)", "N(HUNU+COL1A1+)/N(HUNU+)", "A(COL1a1)/N(HUNU+)"]

            
    im_name = key
    if 
    n = 18
    im_id = a = [im_name[i:i+n] for i in range(0, len(im_name), n)] #extracts the animal id and the code of the image
    im_id = im_id[0]
    
    if im_id in col1a1_names:
        
    match_coloc_hunu_col1 = list(filter(lambda x: im_id in x, col1a1_names))



for filename in filenames:


df = pd.DataFrame(list(zip(filenames, h_c_list, coloc_list, unit_list)))
df.columns = ["Total # of HuNu cells", "# of Hunu cells colocalised with COL1A1", "N(HUNU)+N(COL1A1+) / N(HUNU+)", "A(HUNU+)A(COL1A1)+ / A(HUNU+)" "A(COL1A1+)/A(HUNU+)", "A(COL1a1)/N(HUNU+)"]

with open("/home/inf-54-2020/experimental_cop/scripts/Info.txt", "w") as out: #write the dataframe into an output file
#write the dataframe into an output file
    df.to_string(out, index=None)
    print('output info file saved!')

#         unit = 1.0 * cell_count_coloc / cell_count_h
#         unit = str(unit)
#         unit_list.append(unit)
#         print(cell_count_h)
#         print(cell_count_coloc)
#         print(unit)

                    
#     #count hunu cells within col1a1
#     labels_coloc = measure.label(coloc_cells) #create a binary format of the numpy array (object or background)
#     cell_count_coloc=labels_coloc.max() #count the number of objects
#     coloc_list.append(cell_count_coloc)
#     #create coloc/#nuclei unit
#     unit = 1.0 * cell_count_coloc / cell_count_h
#     unit = str(unit)
#     unit_list.append(unit)
#     print(cell_count_h)
#     print(cell_count_coloc)
#     print(unit)


                    
                    
# for hunu_cells_path, coloc_cells_path in dictionary.items():
#     hunu_cells = cv2.imread(hunu_cells_path)
#     coloc_cells = cv2.imread(coloc_cells_path)
#     filename_h = os.path.basename(hunu_cells_path)
#     filename_c = os.path.basename(coloc_cells_path)
#     print(filename_h)
#     print(filename_c)

#     #count all hunu nuceli
#     labels_h = measure.label(hunu_cells) #create a binary format of the numpy array (object or background)
#     cell_count_h=labels_h.max() #count the number of objects
#     h_c_list.append(cell_count_h)
#     #count hunu cells within col1a1
#     labels_coloc = measure.label(coloc_cells) #create a binary format of the numpy array (object or background)
#     cell_count_coloc=labels_coloc.max() #count the number of objects
#     coloc_list.append(cell_count_coloc)
#     #create coloc/#nuclei unit
#     unit = 1.0 * cell_count_coloc / cell_count_h
#     unit = str(unit)
#     unit_list.append(unit)
#     print(cell_count_h)
#     print(cell_count_coloc)
#     print(unit)


# filenames = ['DAB15sec','DAB30sec','DAB120s']

# df = pd.DataFrame(list(zip(filenames, h_c_list, coloc_list, unit_list)))
# df.columns = ["Filename", "Total # of HuNu cells", "# of Hunu cells colocalised with COL1A1", "#HUNU+COL1A1+ / HUNU+"]

# with open("/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Info.txt", "w") as out: #write the dataframe into an output file
# #write the dataframe into an output file
#     df.to_string(out, index=None)
#     print('output info file saved!')

# # col_start = ["col_a", "col_b", "col_c"]
# # col_add = ["Col_d", "Col_e", "Col_f"]
# # a = pd.DataFrame(list(zip(col_start, col_add)))
# # a.columns = ["Filename", "Total # of HuNu cells", "# of Hunu cells colocalised with COL1A1", "#HUNU+COL1A1+ / HUNU+"]
