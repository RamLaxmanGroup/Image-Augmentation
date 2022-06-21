"""
This script helps when all the images in the source directory 
are not annotated.It selects the image-annotations file pairs.
"""

def xml_for_image(filenames_images, filenames_annotations):
    tmp_img_name, tmp_annot_name = [], []
    for filename in filenames_images:
        filename_ = filename.split('.')[:-1][0] # removes '.jpg' extension and stores the filename
        filename_ += '.xml'
        if filename_ in filenames_annotations:
            tmp_img_name.append(filename)
            tmp_annot_name.append(filename_)
        else:
            print(f"{filename} doesn't have annotation file.")
    
    return tmp_img_name, tmp_annot_name
