"""TO DO"""
"""
Generate the ouput directory using aug_version and class name CLI argument explicitelt
--output_dir ./data/outputs/v1/50  <----- Provided from CLI
It should be generated from --output_dir ./data/outputs --aug_version v1 --class_name class_name

--key AdditiveGaussianNoise  For applying specifie augmentaiton from CLI
"""

import argparse
import cv2
import os
import random
import string
import sys
import numpy as np

from core.augmentation import apply_augmentations

letters = string.ascii_letters

_HELP_TEXT = """
Usage Example:
python run_augmentation.py --input_dir ./data/inputs/v1/base/50 --output_dir ./data/outputs/v1/50 --use_range True --aug_version v1 --start 0.007 --end 0.07 --num_annot 6 ---->  images generated for class '50'
python run_augmentation.py --input_dir ./data/inputs/v1/base/50 --output_dir ./data/outputs/v1/50 --use_range False --aug_version v1 --value 0.07

python run_augmentation.py --input_dir ./data/inputs/v2/base/80 --output_dir ./data/outputs/v1/test --use_range True --aug_version test_v1 --start -5 --end 4 --num_annot 3 --use_angle True
"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description="augment the dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP_TEXT)
    parser.add_argument("--input_dir", help="directory to images to augment.")
    parser.add_argument("--output_dir", help="directory for saving augmented images.")
    parser.add_argument('--value', help="Argument value for child augmenters")
    parser.add_argument("--use_angle", type=str2bool, nargs='?', const=True, default=False, help="True if start and end point are to be given as argument")
    parser.add_argument("--num_annot", default="1", help="number of annotations per image.")
    parser.add_argument("--use_range", type=str2bool, nargs='?', const=True, default=False, help="True if start and end point are to be given as argument")
    parser.add_argument("--start", help="starting value of argument")
    parser.add_argument("--end", help="ending value of argument")
    parser.add_argument("--aug_version", help="Augmentation version")
    parser.add_argument("--class_name", help="Name of class for classification.")
    
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
        #print("Cannot overwrite data")
        #sys.exit()


def update_filename(filename, aug_version, iter_num=None):
    filename_split = filename.split('.')
    random_suffix = str(random.randint(100, 999)) + ''.join((random.choice(letters) for _ in range(3))) + str(random.randint(100, 999))
    
    if iter_num:
        new_filename = aug_version+ "_" + filename_split[0] + "_" + str(iter_num) + "_" + random_suffix + ".jpg"
    else:
        new_filename = aug_version+ "_" + filename_split[0] + "_" + random_suffix + ".jpg"
    return new_filename


def annotation_with_range(input_dir, output_dir, num_annot, start, end, aug_version, use_angle):
    #print(type(input_dir), type(output_dir), type(num_annot), type(start), type(end))

    filenames_images = os.listdir(input_dir)         
    #filenames_images = [filenames_images[0]]     # for train_v2 version of analog meter 

    # determine the argument values for each child annotations
    arg_values = np.linspace(start=start , stop=end, num=num_annot)
    arg_values = [round(val, 3) for val in arg_values]
    #print(arg_values)

    for filename in filenames_images:
        img = cv2.imread(os.path.join(input_dir, filename))
        
        for iter_num, arg_value in zip(range(num_annot), arg_values):
            transformation = apply_augmentations(value=arg_value, use_angle=use_angle)
            augmented_img = transformation(image=img)

            new_filename = update_filename(filename=filename, aug_version=aug_version, iter_num=iter_num)
            new_imagepath = os.path.join(output_dir, new_filename)
            #print(new_filename, new_imagepath)

            cv2.imwrite(new_imagepath, augmented_img)



def annotation_without_range(input_dir, output_dir, value, aug_version):
    #print(type(input_dir), type(output_dir), type(value))
    filenames_images = os.listdir(input_dir)
    transformation = apply_augmentations(value=value)
    for filename in filenames_images:
        img = cv2.imread(os.path.join(input_dir, filename))
        augmented_img = transformation(image=img)

        new_filename = update_filename(filename=filename, aug_version=aug_version)
        new_imagepath = os.path.join(output_dir, new_filename)
        #print(new_filename, new_imagepath)

        cv2.imwrite(new_imagepath, augmented_img)
        


if __name__=="__main__":
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    value = args.value
    use_angle = args.use_angle
    num_annot = args.num_annot
    use_range = args.use_range
    start = args.start
    end = args.end
    aug_version = args.aug_version
    class_name = args.class_name

    
    if use_range:
        if type(start) == type(None):
            print("Start value of range is not provided.")
            sys.exit()
        elif type(end) == type(None):
            print("End value of range is not provided.")
            sys.exit()
        elif type(num_annot) == type(None):
            print("Number of annotation value is not provided.")
            sys.exit()
        else:
            start = float(start)
            end = float(end)
            num_annot = int(num_annot)
    else:
        if type(value) == type(None):
            print("Value for augmenter is not provided.")
            sys.exit()
        else:
            value = float(value)
    
    # Generate ouptut_dir
    
    check_make_dir(output_dir)

    if not check_dir(input_dir) or not check_dir(output_dir):
        print("Input or output directory missing.")
    if use_range:
        annotation_with_range(input_dir, output_dir, num_annot, start, end, aug_version, use_angle)
    else:
        annotation_without_range(input_dir, output_dir, value, aug_version)
        
        

