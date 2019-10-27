import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.io import loadmat, savemat
import scipy


sys.path.append('../')
from preprocColorSpace import preprocColorSpace
from preprocWavelets_grid import preprocWavelets_grid

sys.path.append('../utils')

from preprocColorSpace_GetMetaParams import preprocColorSpace_GetMetaParams
from preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams
from preprocNonLinearOut_GetMetaParams import preprocNonLinearOut_GetMetaParams
from preprocDownsample_GetMetaParams import preprocDownsample_GetMetaParams
from preprocNormalize_GetMetaParams import preprocNormalize_GetMetaParams
from preprocNonLinearOut import preprocNonLinearOut
from preprocDownsample import preprocDownsample
from preprocNormalize import preprocNormalise


do_followup_viz = True

#To run this demo, you will need a stack of movie frames stored as a 4D
# (X x Y x Color x Time) array.
# If you have your own stimuli, you will need to modify this variable:

fname = 'nishimoto_2011_val_1min_uint8.mat'


#load image
try:
    d = h5py.File(fname)
    S = d['S'].value
    S = S.transpose(3,2,1,0)
    d.close()

except:
    print(
        'You may need to modify the "fname" variable in ComputeMotionEnergy.py\n'
        'to point to a .mat file with movie frames in it!\n'
        'A sample file can be downloaded at https://www.dropbox.com/s/1531dr5u7767wat/nishimoto_2011_val_1min_uint8.mat?dl=0\n')

#the field d.S is an array that is (96 x 96 x 3 x 900); (X x Y x Color x
#Images).  The images are stored as 8-bit integer arrays (no decimal
#places, with pixel values from 0-255). These should be converted to
#floating point decimals from 0-1:

S = S.astype(np.float)/ 255


## Preprocessing
# Conver to grayscale (luminance only)
# The argument 1 here indicates a pre-specified set of parameters to feed
# to the preprocColorSpace function to convert from RGB images to
# keeping only the luminance channel. (You could also use matlab's
# rgb2gray.m function, but this is more principled.) Inspect cparams to see
# what those parameters are.
cparams = preprocColorSpace_GetMetaParams(argNum= 1);
S_lum, cparams = preprocColorSpace(S, cparams);


## Gabor wavelet processing
# Process with Gabor wavelets
# The numerical argument here specifies a set of parameters for the
# preprocWavelets_grid function, that dictate the locations, spatial
# frequencies, phases, and orientations of Gabors to use. 2 specifies Gabor
# suitable for computing motion energy in movies.
gparams = preprocWavelets_gird_GetMetaParams(argNum=1); #2 is large
[S_gab, gparams] = preprocWavelets_grid(S_lum, gparams);

## Optional additions
# Compute log of each channel to scale down very large values
nlparams = preprocNonLinearOut_GetMetaParams(argNum=1);
[S_nl, nlparams] = preprocNonLinearOut(S_gab, nlparams);

# Downsample data to the sampling rate of your fMRI data (the TR)
dsparams = preprocDownsample_GetMetaParams(argNum=1); # for TR=1; use (2) for TR=2
[S_ds, dsparams] = preprocDownsample(S_nl, dsparams);

# Z-score each channel
nrmparams = preprocNormalize_GetMetaParams(argNum= 1);
[S_fin, nrmparams] = preprocNormalise(S_ds, nrmparams);


