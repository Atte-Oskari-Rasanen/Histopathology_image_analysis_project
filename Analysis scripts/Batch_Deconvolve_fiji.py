from ij import IJ, ImagePlus
import random
import os
import ntpath

def im_id():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

uniq_id = im_id()

SourceDir = IJ.getDirectory("Choose Source Directory ");

#TargetDir = IJ.getDirectory("Choose Destination Directory ");

#list = IJ.getFileList(SourceDir);
patch_size = 736

main_dir = '/home/atte/Pictures/deconv'
try:
    os.mkdir(main_dir)
except OSError:
    print('Attempted to create a directory for deconvolved images but one already exists')
    pass


    

for root, subdirectories, files in os.walk(SourceDir):
    print('root:' + str(root))
    print('subdirs' + str(subdirectories))
    print('files' + str(files))
    for subdir in subdirectories:
        print(subdir)
        animal_dir = main_dir + '/' + subdir
        #under Deconvolved_im/ make a subdir for each image id
        try:
            os.mkdir(animal_dir)
        except OSError:
            print ("Creation of the directory %s failed" % animal_dir)
        else:
            print ("Successfully created the directory %s " % animal_dir)

        hunu_ch_dir = subdir.rsplit('/')[-1] + '_hunu_ch'
        hunu_ch_dir = animal_dir + hunu_ch_dir
        col1a1_ch_dir = subdir.rsplit('/')[-1] + '_col1a1_ch'
        col1a1_ch_dir = animal_dir + col1a1_ch_dir
        print(hunu_ch_dir)
        print(col1a1_ch_dir)
        try:
            os.mkdir(hunu_ch_dir)
        except OSError:
            print ("Creation of the directory %s failed" % hunu_ch_dir)
        else:
            print ("Successfully created the directory %s " % hunu_ch_dir)
        
        print(col1a1_ch_dir)
        try:
            os.mkdir(col1a1_ch_dir)
        except OSError:
            print ("Creation of the directory %s failed" % col1a1_ch_dir)
        else:
            print ("Successfully created the directory %s " % col1a1_ch_dir)
        subdir_path = str(root) + str(subdir)

        print('subdir path:' + str(subdir_path)) + '/'
        #print(glob.glob(subdir_path))
        im_index = 0
        for file in os.listdir(subdir_path):
	        # print('file path:' )
	        # print(os.path.join(subdir, file))
	        print('file:'+file)
	        imagepath=subdir_path + '/' + file
	        if '~' in imagepath:
	            imagepath = imagepath.split('~')[0]
	        print('imagepath: '+ imagepath)
	        # if not imagefile.endswith('.tif') or imagefile.endswith('.jpg'): #exclude files not ending in .tif
	        #     continue
	        #print(imagepath)
	        imagename_suffix=ntpath.basename(imagepath)#take the name of the file from the path and save it
	        imagename = imagename_suffix.split('.')[0]
	        id_hunu_col = next(uniq_id) # specific id for the corresponding hunu and its col1a1 image		
	        new_imagename = str(im_index) + '_' +str(id_hunu_col) +'_'+ subdir + '_' + str(patch_size)
	        im_index +=1
	        print("imagename: " + imagename)
			###### running the deconvolution step using imagej 
	        IJ.open(imagepath)
	        #imp = IJ.getImage(imagepath)

	        #title = IJ.getTitle();
	        IJ.run("Colour Deconvolution", "vectors=[H DAB] show")
	        print("Processing: " + imagename_suffix)
	        
	        IJ.selectWindow(imagename_suffix + "-(Colour_1)")
	        IJ.run("Enhance Contrast...", "saturated=0.3 normalize equalize")
	        IJ.run("Non-local Means Denoising", "sigma=1 smoothing_factor=1 auto")
	        IJ.saveAs("TIFF",hunu_ch_dir+ '/' + new_imagename + "_HUNU.tif")
	        IJ.selectWindow(imagename_suffix + "-(Colour_3)")
	        IJ.run("Enhance Contrast...", "saturated=0.3 normalize equalize")
	        IJ.run("Non-local Means Denoising", "sigma=1 smoothing_factor=1 auto")
	        IJ.saveAs("TIFF",col1a1_ch_dir+ '/'+ new_imagename + "_COL1A1.tif")
	        #IJ.close(imagepath)
			#im_index += 1
			#print imagename
print('Deconvolution done! Starting segmentation...')
