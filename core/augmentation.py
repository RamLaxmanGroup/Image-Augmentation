from cv2 import rotate
from matplotlib import scale
from numpy import angle
from imgaug import augmenters

def get_child_augmenters():
    child_augment_dict = dict()

    #child_augment_dict['AdditiveGaussianNoise'] = augmenters.arithmetic.AdditiveGaussianNoise(scale=0.05*255)
    #child_augment_dict['AdditiveLaplaceNoise'] = augmenters.arithmetic.AdditiveLaplaceNoise(scale=0.05*255)
    #child_augment_dict['AdditivePoissonNoise'] = augmenters.arithmetic.AdditivePoissonNoise(lam=(0.0, 15.0))

    #child_augment_dict['CoarseDropout'] = augmenters.arithmetic.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.15))
    #child_augment_dict['Dropout'] = augmenters.arithmetic.Dropout((0.0, 0.05))
    
    #child_augment_dict['AverageBlur'] = augmenters.blur.AverageBlur(k=(2, 5))
    #child_augment_dict['GaussianBlur'] = augmenters.blur.GaussianBlur(sigma=(0.1, 2.0))
    #child_augment_dict['MotionBlur'] = augmenters.blur.MotionBlur(k=(4,8), angle=(0,360), direction=(-1, 1))
    #child_augment_dict['BilateralBlur'] = augmenters.blur.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
    
    #child_augment_dict['Rain'] = augmenters.weather.Rain(speed=(0.1,0.3))
    #child_augment_dict['Snowflakes'] = augmenters.weather.Snowflakes(flake_size=(0.2, 0.6), speed=(0.01, 0.05))
    
    #child_augment_dict['GammaContrast'] = augmenters.contrast.GammaContrast((0.5, 1.8))
    #child_augment_dict['SigmoidContrast'] = augmenters.contrast.SigmoidContrast(gain=(4, 6), cutoff=(0.3, 0.6))
    #child_augment_dict['LogContrast'] = augmenters.contrast.LogContrast(gain=(0.6, 1.4), per_channel=False)
    #child_augment_dict['LinearContrast'] = augmenters.contrast.LinearContrast((0.4, 1.6), per_channel=False)
    #child_augment_dict['HistogramEqualization'] = augmenters.contrast.HistogramEqualization()
    #child_augment_dict['Alpha'] = augmenters.Alpha((0.0, .50), augmenters.contrast.HistogramEqualization())
    
    #child_augment_dict['Fliplr'] = augmenters.Fliplr(0.5) # Flips 50% of images Horizontal flip
    #child_augment_dict['Flipud'] = augmenters.Flipud(0.5)

    #child_augment_dict['PerspectiveTransform'] = augmenters.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True,  fit_output=True)
    #child_augment_dict['ElasticTransformation'] = augmenters.geometric.ElasticTransformation(alpha=(0.0, 10.0), sigma=5.0), # use it rarely
    #child_augment_dict['Affine'] = augmenters.Affine(scale=(0.3, 1.3), rotate=(180),fit_output=True)
    child_augment_dict['Affine'] = augmenters.Affine(scale=0.2, fit_output=True) # zoom out on images by 5 times (1/5 =0.2)
    #child_augment_dict['ScaleX'] = augmenters.geometric.ScaleX(scale=(0.5, 1.5), fit_output=True)
    #child_augment_dict['ScaleX'] = augmenters.geometric.ScaleX(scale=(0.5, 1.5), fit_output=True)

    return child_augment_dict


'''MODIFY THE AUGMENTATIONS BASED ON THE REQUIREMENTS'''
def apply_augmentations():
    child_augment_dict = get_child_augmenters()

    transformations_someof = augmenters.SomeOf(n=(1, 9), children=list(child_augment_dict.values()))

    trans = augmenters.Sequential([
        #augmenters.arithmetic.AdditiveGaussianNoise(scale=0.05*255),
        #augmenters.arithmetic.AdditiveLaplaceNoise(scale=0.05*255),
        #augmenters.arithmetic.AdditivePoissonNoise(lam=(0.0, 15.0)),

        #augmenters.arithmetic.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.5)),
        #augmenters.arithmetic.Dropout((0.0, 0.05)),
        
        #augmenters.blur.AverageBlur(k=(2, 5)),
        #augmenters.blur.GaussianBlur(sigma=(0.0, 3.0)),
        #augmenters.blur.MotionBlur(k=7),
        #augmenters.blur.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
        
        #augmenters.weather.Rain(speed=(0.5,0.9)),
        #augmenters.weather.Snowflakes(flake_size=(0.2, 0.6), speed=(0.01, 0.05)),
        
        #augmenters.contrast.GammaContrast((0.5, 1.8)),
        #augmenters.contrast.SigmoidContrast(gain=(4, 6), cutoff=(0.4, 0.6)),
        #augmenters.contrast.LogContrast(gain=(0.6, 1.4), per_channel=False),
        #augmenters.contrast.LinearContrast((0.4, 1.6), per_channel=False),
        #augmenters.contrast.HistogramEqualization(),
        #augmenters.Alpha((0.0, .50), augmenters.contrast.HistogramEqualization()),
        
        #augmenters.Fliplr(0.5), # Flips 50% of images
        #augmenters.Flipud(0.5),

        #augmenters.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True,  fit_output=True),
        #augmenters.geometric.ElasticTransformation(alpha=(0.0, 10.0), sigma=5.0), # use it rarely
        #augmenters.Affine(
        #    scale=(0.3, 1.3),
        #    fit_output=True
        #)
        #augmenters.geometric.ScaleX(scale=(0.5, 1.5), fit_output=True),
        #augmenters.geometric.ScaleX(scale=(0.5, 1.5), fit_output=True),
        #augmenters.WithPolarWarping(augmenters.CropAndPad(percent=(-0.1, 0.1))), # Don't use
        
    ])
    return transformations_someof