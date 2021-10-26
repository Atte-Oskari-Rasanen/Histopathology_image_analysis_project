import cv2
from skimage import measure, filters
import os 
import pandas as pd
ws_coloc_path = ['/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Coloc_DAB_15s_hunu_segm.png',
                '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Coloc_DAB_30s_hunu_segm.png',
                '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Coloc_DAB_120s_hunu_segm.png'] 
ws_hunu_path = ['/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/DAB_15s_hunu_segm.pngoutlines.png',
                '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/DAB_30s_hunu_segm.pngoutlines.png',
                '/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/DAB_120s_hunu_segm.pngoutlines.png'] 

dictionary = dict(zip(ws_hunu_path, ws_coloc_path))

'/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/'

h_c_list = []
coloc_list = []
unit_list = []
for hunu_cells_path, coloc_cells_path in dictionary.items():
    hunu_cells = cv2.imread(hunu_cells_path)
    coloc_cells = cv2.imread(coloc_cells_path)
    filename_h = os.path.basename(hunu_cells_path)
    filename_c = os.path.basename(coloc_cells_path)
    print(filename_h)
    print(filename_c)

    #count all hunu nuceli
    labels_h = measure.label(hunu_cells) #create a binary format of the numpy array (object or background)
    cell_count_h=labels_h.max() #count the number of objects
    h_c_list.append(cell_count_h)
    #count hunu cells within col1a1
    labels_coloc = measure.label(coloc_cells) #create a binary format of the numpy array (object or background)
    cell_count_coloc=labels_coloc.max() #count the number of objects
    coloc_list.append(cell_count_coloc)
    #create coloc/#nuclei unit
    unit = 1.0 * cell_count_coloc / cell_count_h
    unit = str(unit)
    unit_list.append(unit)
    print(cell_count_h)
    print(cell_count_coloc)
    print(unit)


filenames = ['DAB15sec','DAB30sec','DAB120s']

df = pd.DataFrame(list(zip(filenames, h_c_list, coloc_list, unit_list)))
df.columns = ["Filename", "Total # of HuNu cells", "# of Hunu cells colocalised with COL1A1", "#HUNU+COL1A1+ / HUNU+"]

with open("/home/inf-54-2020/experimental_cop/batch6/batch6_processed_ws/Info.txt", "w") as out: #write the dataframe into an output file
#write the dataframe into an output file
    df.to_string(out, index=None)
    print('output info file saved!')
