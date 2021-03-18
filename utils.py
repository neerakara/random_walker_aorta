import numpy as np
from scipy.ndimage import gaussian_filter

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