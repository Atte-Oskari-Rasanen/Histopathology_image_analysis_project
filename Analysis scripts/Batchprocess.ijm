//INPUT = getArgument();

INPUT = "/home/atte/Documents/PD_images/batch8_retry"
//INPUT = "/home/atte/Documents/PD_images/20"
//INPUT = "/home/inf-54-2020/experimental_cop/batch8_redo"

//#@input
//#@output

print(INPUT);


//#@ File (label = "Input directory", style = "directory") input
//#@ File (label = "Output directory", style = "directory") output
//#@ String (label = "File suffix", value = ".tif") suffix
setBatchMode(true);
// See also Process_Folder.py for a version of this code
// in the Python scripting language.
suffix = ".tif"
processFolder(INPUT);


//hunu_dir_unproc = input + File.separator + HUNU_unproc;
//hunu_dir_proc = input + File.separator + HUNU_proc;
//col1a1_dir_unproc = input + File.separator + COL1A1_unproc;
//col1a1_dir_proc = input + File.separator + COL1A1_proc;
// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(INPUT) {
	list = getFileList(INPUT);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(INPUT + File.separator + list[i]))
			processFolder(INPUT + File.separator + list[i]);
			print(list[i]);
		if(endsWith(list[i], suffix))
			processFile(INPUT, list[i]);
	}
}


function processFile(INPUT, file) {
		//for (i = 0; i < 12; i++) {
	path = INPUT + File.separator +file;
	open(path);
    title=getTitle();
		//run("Colour Deconvolution", "vectors=[H DAB] show");
	run("Colour Deconvolution", "vectors=[User values] [r1]=0.6500286 [g1]=0.704031 [b1]=0.2860126 [r2]=0.515 [g2]=0.639 [b2]=0.571 [r3]=0.268 [g3]=0.570 [b3]=0.7764");
	print("Processing: " + INPUT + File.separator + file);
	print("Saving to: " + INPUT);
	//run("Enhance Contrast", "saturated=0.35");
	//run("Apply LUT");

	//Save the channels
   	selectWindow(title + "-(Colour_3)");
   	run("Median...", "radius=2");
	//run("Enhance Contrast...", "saturated=0.3 normalize equalize");
    run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None*");
   	saveAs("PNG",INPUT+ File.separator+replace(title,suffix,"_hunu.png"));
	close(title + "-(Colour_3)");
	close(path);
	path = INPUT + File.separator +file;
	open(path);
    title=getTitle();
	//for col1a1 using another deconvolution method and removing the nuclei afterwards:

    selectWindow(title + "-(Colour_1)");
    run("Median...", "radius=2");
	//run("Enhance Contrast...", "saturated=0.3 normalize equalize");
	run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None*");
    
    saveAs("PNG",INPUT+ File.separator+replace(title,suffix,"_col1a1.png"));
    close(title + "-(Colour_1)");
    close(path);
}
