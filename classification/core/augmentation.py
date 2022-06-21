from tracemalloc import start
from imgaug import augmenters

def get_child_augmenters(aug_name, value=None, use_angle=None):#, start=None, end=None):
    child_augment_dict = dict()
    if (value != None) and (use_angle != None):
        if use_angle:
            print("Value as angle given. \n")
            child_augment_dict['Rotate'] = augmenters.geometric.Rotate(rotate=value, fit_output=True)

        else:
            """APPLY CONDITION FOR SELECTING CHILD AUGMENTERS FOR CERTAIN 'value'""" # {TO DO}
            # Implementaion of exception handling in case of invalid value  {TO DO}
            if aug_name == 'AdditiveGaussianNoise':
                child_augment_dict['AdditiveGaussianNoise'] = augmenters.arithmetic.AdditiveGaussianNoise(scale=value*255)
            elif aug_name == 'AdditiveLaplaceNoise':
                child_augment_dict['AdditiveLaplaceNoise'] = augmenters.arithmetic.AdditiveLaplaceNoise(scale=value*255)
            #child_augment_dict['AdditivePoissonNoise'] = augmenters.arithmetic.AdditivePoissonNoise(lam=(0.0, 15.0))

            #child_augment_dict['CoarseDropout'] = augmenters.arithmetic.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.15))
            #child_augment_dict['Dropout'] = augmenters.arithmetic.Dropout((0.0, 0.05))
            
            #child_augment_dict['AverageBlur'] = augmenters.blur.AverageBlur(k=(2, 5))
            elif aug_name == 'GaussianBlur':
                child_augment_dict['GaussianBlur'] = augmenters.blur.GaussianBlur(sigma=value)
            #child_augment_dict['MotionBlur'] = augmenters.blur.MotionBlur(k=value, angle=(0,360), direction=(-1, 1))
            #child_augment_dict['BilateralBlur'] = augmenters.blur.BilateralBlur(d=value, sigma_color=(10, 250), sigma_space=(10, 250))
            
            #child_augment_dict['Rain'] = augmenters.weather.Rain(speed=value)
            #child_augment_dict['Snowflakes'] = augmenters.weather.Snowflakes(flake_size=value, speed=(0.01, 0.05))
            elif aug_name == 'GammaContrast':
                child_augment_dict['GammaContrast'] = augmenters.contrast.GammaContrast(value)
            #child_augment_dict['SigmoidContrast'] = augmenters.contrast.SigmoidContrast(gain=(4, 6), cutoff=(0.3, 0.6))
            elif aug_name == 'LogContrast':
                child_augment_dict['LogContrast'] = augmenters.contrast.LogContrast(gain=value, per_channel=False)
            elif aug_name == 'LinearContrast':
                child_augment_dict['LinearContrast'] = augmenters.contrast.LinearContrast(value, per_channel=False)
            #child_augment_dict['HistogramEqualization'] = augmenters.contrast.HistogramEqualization()
            #child_augment_dict['Alpha'] = augmenters.Alpha((0.0, .50), augmenters.contrast.HistogramEqualization())
            
            #child_augment_dict['Fliplr'] = augmenters.Fliplr(value) # Flips 50% of images Horizontal flip
            #child_augment_dict['Flipud'] = augmenters.Flipud(value)

            #child_augment_dict['PerspectiveTransform'] = augmenters.PerspectiveTransform(scale=value, keep_size=True,  fit_output=True)
            #child_augment_dict['ElasticTransformation'] = augmenters.geometric.ElasticTransformation(alpha=value, sigma=5.0), # use it rarely
            #child_augment_dict['Affine'] = augmenters.Affine(scale=value, rotate=(90, 180), fit_output=True) # zoom out on images by 5 times (1/5 =0.2)
            elif aug_name == 'ScaleX':
                child_augment_dict['ScaleX'] = augmenters.geometric.ScaleX(scale=value, fit_output=True)
            elif aug_name == 'ScaleY':
                child_augment_dict['ScaleY'] = augmenters.geometric.ScaleY(scale=value, fit_output=True)

    return child_augment_dict
    


'''MODIFY THE AUGMENTATIONS BASED ON THE REQUIREMENTS'''
def apply_augmentations(aug_name, value=None, use_angle=None):#, start=None, end=None):
    # augmenters.SomeOf
    # augmenters.Sequential
    print(f"{aug_name} applied to augmenters with value {value}")#, "start: ", start, "end: ", end)

    child_augment_dict = get_child_augmenters(aug_name, value, use_angle)#, start, end)
    #print("elements in dictionary: ", len(child_augment_dict))

    if not bool(child_augment_dict): # return False only when dict is empty
        raise ValueError("Cannot get child augmenters. Did you miss to supply value for argument 'value'?")

    #transformations_someof = augmenters.SomeOf(n=(1, 9), children=list(child_augment_dict.values()))

    if value:
        transformations_seq = augmenters.Sequential([
            child_augment_dict[aug_name]
        ])
        return transformations_seq