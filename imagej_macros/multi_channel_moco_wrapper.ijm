// Use this code to run moco on multi channel data
currently broken: Python cannot read the metadata from the files
getDimensions(w, h, channels, slices, frames);
// tolerate time stacks and z stacks
if (slices<frames) {
	slices=frames;
}
title = getTitle();
// make a copy that will be used to calculate transformations
run("Duplicate...", "title=high_contrast duplicate");
// moco is supposed to work better with 8 bit images
run("8-bit");
// Also with high contrast images
run("Normalize Local Contrast", "block_radius_x="+w+" block_radius_y="+h+" standard_deviations=3 center stretch stack");
selectWindow("high_contrast");
// make the template
run("Make Substack...", "  slices="+floor(slices/2));
template_title = getTitle();
// run moco on the 8 bit high contrast image saving the results
run("moco ", "value=51 downsample_value=1 template=["+template_title+"] stack=high_contrast log=[Generate log file] plot=[No plot]");
// We no longer need the high contrast images or the corrected image. Just the results
selectWindow(template_title);
close();
selectWindow("high_contrast");
close();
selectWindow("New Stack");
close();
selectWindow(title);
// do transformation on the original image
for (i = 0; i < nResults; i++) {
	setSlice(i+1);
	x = getResult("x", i);
	y = getResult("y", i);
	run("Translate...", "x="+x+" y="+y+" interpolation=None slice");
}
// change the name of the corrected file, and make a z projection
rename("UNDRIFTED"+title);
run("Z Project...", "projection=[Max Intensity]");

