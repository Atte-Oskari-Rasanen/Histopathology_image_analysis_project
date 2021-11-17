
setBatchMode(true);
// See also Process_Folder.py for a version of this code
// in the Python scripting language.


input = "/home/atte/Documents/PD_images/batch8_retry/Deconvolved_ims";
//input = "/home/atte/Documents/PD_images/batch8_retry";
//input = "/home/atte/Desktop/Testing_coloc/hunu_th"
size = "300-3500";
suffix = "Segm_TH.png";
//#@ File (label = "Input directory", style = "directory") input
//#@ String (label = "File suffix", value = ".Semg_TH.png") suffix
// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
			print(list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, list[i]);
	}
}
print(input);

function processFile(input, file) {
		//for (i = 0; i < 12; i++) {
	path = input + File.separator + file;
	open(path);
    title=getTitle();
	//run("Bio-Formats Importer", "open=" + path + " autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
	
	run("Invert LUT");
	
	run("Analyze Particles...", "size=300-4000 show=Masks");
	run("Watershed");
	selectWindow("Mask of " + title);

	print("Processing: " + input + File.separator + title);
	print("Saving to: " + input);
	//run("Invert LUT");
	//run("Enhance Contrast", "saturated=0.35");
	//run("Apply LUT");
	saveAs("PNG",input+ File.separator+replace(title,suffix,"_WS.png"));
   	//saveAs("TIFF",input+File.separator+file);
    close(path);
}


processFolder(input);

print("WATERSHED DONE!");
