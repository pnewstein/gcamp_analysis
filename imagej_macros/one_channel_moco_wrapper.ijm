
// Use this code to run moco on a 1 channel calcium imaging file
getDimensions(w, h, channels, slices, frames);
// tolerate time stacks and z stacks
if (slices<frames) {
	slices=frames;
}
title = getTitle();
metadata = getMetadata("Info");
// get the template stack
run("Make Substack...", "  slices="+floor(slices/2));
template_title = getTitle();
// run moco
run("moco ", "value=51 downsample_value=1 template=["+template_title+"] stack="+title+" log=None plot=[No plot]");
// make a Z projection
run("Z Project...", "projection=[Max Intensity]");
// copy over the metadata
selectWindow("New Stack");
setMetadata("Info", metadata);
rename("UNDRIFTED"+title);
// close the template
selectWindow(template_title);
close();
