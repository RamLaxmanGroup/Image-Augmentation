
import argparse
import cv2
import os
import random
import string
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from core.utils import read_xml, bbox_in_char, extract_coordinates
from core.augmentation import apply_augmentations
from core.utils import float_coordinates_to_int, write_xml
from core.select_xml_for_image import xml_for_image


_HELP_TEXT = """
Usage Example:
python run_augmentation.py --
"""

def get_args():
    parser = argparse.ArgumentParser(description='Augment your image dataset',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog=_HELP_TEXT)
    parser.add_argument("--input_images_path", help="input image/s directory.")
    parser.add_argument("--input_annotations_path", help="input annotation/s directory.")
    parser.add_argument("--output_images_path", default="./dataset/images", help="output image/s directory.")
    parser.add_argument("--output_annotations_path", default="./dataset/annotations", help="output annotation/s directory.")

    args = parser.parse_args()
    args.shape_override = None
    return args



def check_dir(path):
    if os.path.exists(path):
        return True

def check_make_dir(path):
    current_path = os.getcwd().replace('\\', '/')
    if not os.path.exists(path):
        path = path.split('/')[1:]
        for i in range(len(path)):
            current_path = os.path.join(current_path, path[i]) # path[:i+1]
            if not os.path.exists(current_path):
                os.mkdir(current_path)
    else:
        print(f'{path} exists:')


def main(input_images_path, input_annotations_path, output_images_path, output_annotations_path, filenames_images, filenames_annotations):
    num_of_annotations = 10
    for filename_image in filenames_images:
        for iter_num in range(num_of_annotations):
            bboxes = []
            filename_split = filename_image.split('.')
            filename_annotation = filename_split[0] + '.xml'
            if filename_annotation in filenames_annotations:
                path_xml_file = os.path.join(input_annotations_path, filename_annotation)
                folder, _, path_img, width, height, depth, segmented, objects = read_xml(path_xml_file)

            else:
                print(filename_annotation + ' doesn\'t exists in directory. \n Check filenames')
                break
            
            letters = string.ascii_letters
            random_suffix = str(random.randint(100, 999)) + ''.join((random.choice(letters) for _ in range(5))) + str(random.randint(1000, 2999))
            new_filename = filename_split[0] + "_" + str(iter_num) + "_" + random_suffix
            

            for object in objects:
                name = object.find('name').get_text()
                pose = object.find('pose').get_text()
                truncated = object.find('truncated').get_text()
                difficult = object.find('difficult').get_text()
                bndbox = object.find('bndbox').get_text()
                coordinates = bbox_in_char(bndbox)
                bboxes.append([name, pose, truncated, difficult, extract_coordinates(coordinates)])
            

            # read bounding boxes
            bbs_lst = []
            for val in bboxes:
                #print(val[4][1])
                bbs_lst.append(BoundingBox(x1=val[4][0], y1=val[4][1], x2=val[4][2], y2=val[4][3]))
            
            # Read image 
            image = cv2.imread(os.path.join(input_images_path, filename_image))

            # Preparing bounding boxes for augmentations
            bbs = BoundingBoxesOnImage(bbs_lst, shape=image.shape)

            # Defing all the augmentaions to be applied
            transformations = apply_augmentations()

            # applying augmentations
            img_augmented, bbs_augmented = transformations(image=image, bounding_boxes=bbs)

            new_filename_image = new_filename + '.jpg'
            new_filename_xml = new_filename + '.xml'
            new_filename_image_path = os.path.join(output_images_path, new_filename_image)
            new_filename_xml_path = os.path.join(output_annotations_path, new_filename_xml)
            #
            # print(new_filename_image_path, new_filename_xml_path)

            # write augmented image in disk
            cv2.imwrite(new_filename_image_path, img_augmented)

            # update the bounding box coordinates in augmented image
            for i in range(len(bbs.bounding_boxes)):
                
                before = bbs.bounding_boxes[i]
                before = float_coordinates_to_int(before)

                after = bbs_augmented.bounding_boxes[i].clip_out_of_image(image.shape)
                after = float_coordinates_to_int(after)
                #print(before.x1, after.x1)
                
                bboxes[i][4][0] = after.x1
                bboxes[i][4][1] = after.y1
                bboxes[i][4][2] = after.x2
                bboxes[i][4][3] = after.y2
            
            # updating path
            path_img = ('\\').join(path_img.split('\\')[:-1])
            path_img = os.path.join(path_img, new_filename_image)
            #print(path_img)

            # write augmented bounded box in xml file
            write_xml(folder, new_filename_image, path_img, width, height, depth, segmented, bboxes, new_filename_xml_path)




if __name__=='__main__':
    args = get_args()

    input_images_path = args.input_images_path
    input_annotations_path = args.input_annotations_path
    output_images_path = args.output_images_path
    output_annotations_path = args.output_annotations_path

    check_make_dir(output_images_path)
    check_make_dir(output_annotations_path)
    

    if check_dir(input_images_path) and check_dir(input_annotations_path):
        filenames_images = os.listdir(input_images_path)
        filenames_annotations = os.listdir(input_annotations_path)
        if len(filenames_images) == len(filenames_annotations):
            main(input_images_path, input_annotations_path, output_images_path, output_annotations_path, filenames_images, filenames_annotations)
        else:
            print('\nAll images don\'t have their annotations')
            print('Choosing images which have their corresponding annotations')
            filenames_images, filenames_annotations = xml_for_image(filenames_images, filenames_annotations)
            print(filenames_images, filenames_annotations)
            main(input_images_path, input_annotations_path, output_images_path, output_annotations_path, filenames_images, filenames_annotations)
            
    else:
        print('Image or Annotation directory is wrong. Please update it with correct one.')