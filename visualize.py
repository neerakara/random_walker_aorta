# Use the RW algorithm to segment in 3D at one time point
# Make the segmentation slightly larger and extend it to all other time points
# Save the segmentated image using the hpc-predict-io classes

# ============================   
# import module and set paths
# ============================   
import numpy as np
from mr_io import FlowMRI, SegmentedFlowMRI
import matplotlib.pyplot as plt
import utils
import imageio


basepath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/code/hpc-predict/'
basepath = basepath + 'data/v1/decrypt/flownet/hpc_predict/v2/inference/'
basepath = basepath + '2021-02-11_19-41-32_daint102'

def read_segmentation(subnum):
    subject_specific_basepath = basepath + '_volN' + str(subnum) + '/output/recon_volN' + str(subnum)
    segmentedflowmripath = subject_specific_basepath + '_vn_seg_rw.h5'
    return SegmentedFlowMRI.read_hdf5(segmentedflowmripath)

# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

segmented_flow_mri = read_segmentation(4)
flowMRI_seg4 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(5)
flowMRI_seg5 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(6)
flowMRI_seg6 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

segmented_flow_mri = read_segmentation(7)
flowMRI_seg7 = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean, np.expand_dims(segmented_flow_mri.segmentation_prob, -1)], axis=-1)      

print(flowMRI_seg4.shape)
print(flowMRI_seg5.shape)
print(flowMRI_seg6.shape)
print(flowMRI_seg7.shape)
    
# save as pngs
t = 3
for z in range(19):
    
    plt.figure(figsize=[20,20])
    plt.subplot(2,4,1); plt.imshow(utils.norm(flowMRI_seg4[:,:,z,t,1], flowMRI_seg4[:,:,z,t,2], flowMRI_seg4[:,:,z,t,3]), cmap='gray')
    plt.subplot(2,4,2); plt.imshow(utils.norm(flowMRI_seg5[:,:,z,t,1], flowMRI_seg5[:,:,z,t,2], flowMRI_seg5[:,:,z,t,3]), cmap='gray')
    plt.subplot(2,4,3); plt.imshow(utils.norm(flowMRI_seg6[:,:,z,t,1], flowMRI_seg6[:,:,z,t,2], flowMRI_seg6[:,:,z,t,3]), cmap='gray')
    plt.subplot(2,4,4); plt.imshow(utils.norm(flowMRI_seg7[:,:,z,t,1], flowMRI_seg7[:,:,z,t,2], flowMRI_seg7[:,:,z,t,3]), cmap='gray')
    plt.subplot(2,4,5); plt.imshow(np.round(flowMRI_seg4[:,:,z,t,4]), cmap='gray'); plt.clim([0,1])
    plt.subplot(2,4,6); plt.imshow(np.round(flowMRI_seg5[:,:,z,t,4]), cmap='gray'); plt.clim([0,1])
    plt.subplot(2,4,7); plt.imshow(np.round(flowMRI_seg6[:,:,z,t,4]), cmap='gray'); plt.clim([0,1])
    plt.subplot(2,4,8); plt.imshow(np.round(flowMRI_seg7[:,:,z,t,4]), cmap='gray'); plt.clim([0,1])
    plt.savefig('/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/pngs_z' + str(z) + '.png')
    plt.close()
    
    
zz_gif = []
for z in range(19):
    zz_gif.append(imageio.imread('/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/pngs_z' + str(z) + '.png'))
imageio.mimsave('/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/pngs.gif', zz_gif, format='GIF', duration=0.5)


    
    
