import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.ndimage.morphology as morph
import skimage.morphology as mp

# function to normalize the input arrays (intensity and velocity) to a range between 0 to 1 and -1 to 1
# magnitude normalization is a simple division by the largest value
# velocity normalization first calculates the largest magnitude and then uses the components of this vector to normalize the x,y and z directions seperately
def normalize_arrays(arrays):
    #dimension of normalized_arrays: 128 x 128 x 20 x 25 x 4
    normalized_arrays = np.zeros((arrays.shape))
    #normalize magnitude channel
    normalized_arrays[...,0] = arrays[...,0]/np.amax(arrays[...,0])
    #normalize velocities
    #calculate the velocity magnitude at each voxel
    velocity_arrays = gaussian_filter(np.array(arrays[...,1:4]),0.5)
    
    velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    #find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    vpercentile =  np.percentile(velocity_mag_array,95)
    velocity_mag_array[velocity_mag_array>vpercentile] = 1.0
    vmax = np.amax(velocity_mag_array)
    
    normalized_arrays[...,1] = velocity_arrays[...,0]
    normalized_arrays[...,2] = velocity_arrays[...,1]
    normalized_arrays[...,3] = velocity_arrays[...,2]
        
    normalized_arrays[normalized_arrays>vmax] = vmax
    normalized_arrays[normalized_arrays<-vmax] = -vmax
    
    normalized_arrays[...,1] /= vmax
    normalized_arrays[...,2] /= vmax
    normalized_arrays[...,3] /= vmax
        
    # print('normalized arrays: max=' + str(np.amax(normalized_arrays)) + ' min:' + str(np.amin(normalized_arrays)))
    
    return normalized_arrays

def norm(x,y,z):
    normed_array = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return normed_array 

# closing operation for postprocessing the segmentation to remove holes from the inside to avoid wrong seeds if used for 4D initialization
# takes a 3D volume and returns a 3D volume where every slice is eroded with a "circular" 3x3 kernel
# the rw data is then eroded and diluted and markers are assigned to the two classes, no markers are placed in the overlap so that the RW algorithm can fill these gaps
def erode_segmentation(labels_3d):
    
    kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool)
    
    labels_3d_binary = np.array(labels_3d, dtype=bool)
    
    closed_seg = morph.binary_closing(labels_3d_binary, structure = kernel)    
    
    eroded_seg = np.zeros(labels_3d.shape)
    for i in range(labels_3d.shape[2]):
        eroded_seg[:, :, i] = mp.thin(closed_seg[:, :, i], max_iter = 3)
    dilated_seg = morph.binary_dilation(closed_seg, structure = kernel, iterations = 3)
           
    markers = np.zeros(labels_3d.shape)
    fg_markers = (np.logical_and(eroded_seg, dilated_seg)) * 1
    bg_markers = (np.logical_and(np.logical_not(markers), np.logical_not(dilated_seg))) * 2
    markers = fg_markers + bg_markers
    
    return markers