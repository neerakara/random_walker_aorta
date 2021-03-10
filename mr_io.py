import h5py
import os
import logging
import numpy as np
from typing import List

from mr_io_locking_utils import LockedFileH5Reader, LockedFileExclusiveH5Writer

class SpatialMRI:
    """MRI datatype for fixed-time scalar measurements in 3d without axis information
    
    instance fields:
    voxel_feature     (np.ndarray(shape=(x_dim,y_dim,z_dim), dtype=float)):           scalar measurement values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    
    class-level fields:
    group_name :  string                                                              path of datasets in HDF5-file
    """
    
    group_name = "spatial-mri"  

    def __init__(self, scalar_feature: np.ndarray):
        # a numpy array
        if not(len(scalar_feature.shape) == 3):
            raise ValueError("scalar_feature must be a 3D-array (instead it has shape {}).".format(scalar_feature.shape))
        self.scalar_feature = scalar_feature
        

    def write_hdf5(self, path: str):
        """Write this MRI to hpc-predict-io HDF5-format at path"""
        # file handling
        if os.path.exists(path):
            raise FileExistsError("Tried to open file %s, which exists already" % path)
    
        # with h5py.File(path, "w") as f:
        #     scalar_feature_transposed = self.scalar_feature.transpose()
        #     grp = f.create_group(SpatialMRI.group_name)
        #     ds = grp.create_dataset("scalar_feature", scalar_feature_transposed.shape, data=scalar_feature_transposed, dtype="float64")
        lock = LockedFileExclusiveH5Writer(path)
        lock.open()
        f = lock.file()
        scalar_feature_transposed = self.scalar_feature.transpose()
        grp = f.create_group(SpatialMRI.group_name)
        ds = grp.create_dataset("scalar_feature", scalar_feature_transposed.shape, data=scalar_feature_transposed, dtype="float64")
        lock.close()


    def read_hdf5(path: str):
        """Read MRI from file at path as hpc-predict-io HDF5-format"""
        # with h5py.File(path, "r") as f:
        #     # here comes the actual deserialization code
        #     return SpatialMRI(scalar_feature=f[SpatialMRI.group_name]["scalar_feature"][()].transpose())
        lock = LockedFileH5Reader(path)
        lock.open()
        f = lock.file()
        # here comes the actual deserialization code
        mri = SpatialMRI(scalar_feature=f[SpatialMRI.group_name]["scalar_feature"][()].transpose())
        lock.close()
        return mri


def validate_spacetime_coordinates(geometry, time):
    if (len(geometry) != 3):
        raise ValueError("geometry must be a list of 3 one-dimensional ndarrays (instead it has {} elements).".format(len(geometry)))       
    for i in range(3):
        if not(len(geometry[i].shape) == 1):
            raise ValueError("geometry[{}] must be a 3D-array (instead it has shape {}).".format(i, geometry[i].shape))
    if not(len(time.shape) == 1):
        raise ValueError("time must be a 1D-array (instead it has shape {}).".format(time.shape))

def validate_spacetime_feature_coordinate_dims(geometry, time, voxel_feature):
    for i in range(3):
        if not(geometry[i].shape[0] == voxel_feature.shape[i]):
            raise ValueError("geometry[{}] and {}-th dimension in voxel_feature have inconsistent shape: {} vs. {}.".format(i, i, geometry[i].shape[0], voxel_feature.shape[i]))        
    if not(time.shape[0] == voxel_feature.shape[3]):
        raise ValueError("time and {}-th dimension in voxel_feature have inconsistent shape: {} vs. {}.".format(3, time.shape[0], voxel_feature.shape[3]))        

def validate_spacetime_scalar_feature(cls, geometry, time, voxel_feature):
    validate_spacetime_feature_coordinate_dims(geometry, time, voxel_feature)
    if not(len(voxel_feature.shape) == 4):
        raise ValueError("Spacetime scalar field must be a 4D-array (instead it has shape {}).".format(voxel_feature.shape))        
        
def validate_spacetime_vector_feature(cls, geometry, time, voxel_feature):
    validate_spacetime_feature_coordinate_dims(geometry, time, voxel_feature)
    if not(len(voxel_feature.shape) == 5):
        raise ValueError("Spacetime vector field must be a 5D-array (instead it has shape {}).".format(voxel_feature.shape))        
    if not(voxel_feature.shape[4] == 3):
        logging.warning("Constructing {} with non-3-dimensional vector field (instead {}-dimensional)".format(cls.__name__, voxel_feature.shape[4]))            
        
def validate_spacetime_matrix_feature(cls, geometry, time, voxel_feature):
    validate_spacetime_feature_coordinate_dims(geometry, time, voxel_feature)
    if not(len(voxel_feature.shape) == 6):
        raise ValueError("Spacetime matrix field must be a 6D-array (instead it has shape {}).".format(voxel_feature.shape))        
    if not(voxel_feature.shape[4] == 3) or not(voxel_feature.shape[5] == 3):
        logging.warning("Constructing {} with non-3x3-dimensional matrix field (instead {}x{}-dimensional)".format(cls.__name__, voxel_feature.shape[4], voxel_feature.shape[5]))                
        

def write_group_attribute(grp, name, value):
    grp.attrs[name] = value
    
def write_space_time_coordinates(grp, geometry, time):
    for i, coord_name in enumerate(["x_coordinates", "y_coordinates", "z_coordinates"]):
        grp.create_dataset(coord_name, geometry[i].shape, data=geometry[i], dtype="float64")
    grp.create_dataset("t_coordinates", time.shape, data=time, dtype=time.dtype)
    
def write_space_time_voxel_scalar_feature(grp, name, voxel_feature):
    voxel_feature_transposed = voxel_feature.transpose((2,1,0,3))
    ds = grp.create_dataset(name, voxel_feature_transposed.shape, data=voxel_feature_transposed, dtype="float64")
    
def write_space_time_voxel_vector_feature(grp, name, voxel_feature):
    voxel_feature_transposed = voxel_feature.transpose((2,1,0,3,4))
    ds = grp.create_dataset(name, voxel_feature_transposed.shape, data=voxel_feature_transposed, dtype="float64")
    
def write_space_time_voxel_matrix_feature(grp, name, voxel_feature):
    voxel_feature_transposed = voxel_feature.transpose((2,1,0,3,5,4))
    ds = grp.create_dataset(name, voxel_feature_transposed.shape, data=voxel_feature_transposed, dtype="float64")

class SpaceTimeMRI:
    """MRI datatype for time-dependent vectorial measurements in 3d including axis information
    
    instance fields:
    geometry         ([np.ndarray(shape=(x_dim,), dtype=float),
                       np.ndarray(shape=(y_dim,), dtype=float),
                       np.ndarray(shape=(z_dim,), dtype=float)]):                     spatial coordinates along each axis
    time              (np.ndarray(shape=(t_dim,), dtype=float)):                      time coordinates
    voxel_feature     (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim,3), dtype=float)):   vectorial measurement values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    
    class-level fields:
    group_name :  string                                                              path of datasets in HDF5-file
    """

    group_name = "space-time-mri"
    
    def __init__(self, geometry: List[np.ndarray], time: np.ndarray, vector_feature: np.ndarray):
        """Voxel-based parameters must be specified in (x,y,z,t,i)-order, Fortran will treat it in (i,t,x,y,z)-order.
           The index i is used as the component index (i.e. between 0..2 for mean and 0..5 for covariance of velocity field)
        """
        validate_spacetime_coordinates(geometry, time)
        validate_spacetime_vector_feature(SpaceTimeMRI, geometry, time, vector_feature)
        self.geometry = geometry
        self.time = time
        self.vector_feature = vector_feature

    def write_hdf5(self, path: str):
        """Write this MRI to hpc-predict-io HDF5-format at path"""
        # file handling
        if os.path.exists(path):
            raise FileExistsError("Tried to open file %s, which exists already" % path)
    
        # with h5py.File(path, "w") as f:
        #     # here comes the actual serialization code (transposition to use Fortran memory layout)
        #     grp = f.create_group(SpaceTimeMRI.group_name)
        #     write_space_time_coordinates(grp, self.geometry, self.time)
        #     write_space_time_voxel_vector_feature(grp, "vector_feature", self.vector_feature)

        lock = LockedFileExclusiveH5Writer(path)
        lock.open()
        f = lock.file()
        # here comes the actual serialization code (transposition to use Fortran memory layout)
        grp = f.create_group(SpaceTimeMRI.group_name)
        write_space_time_coordinates(grp, self.geometry, self.time)
        write_space_time_voxel_vector_feature(grp, "vector_feature", self.vector_feature)
        lock.close()

    def read_hdf5(path: str):
        """Read MRI from file at path as hpc-predict-io HDF5-format"""
        # with h5py.File(path, "r") as f:
        #     # here comes the actual deserialization code
        #     return SpaceTimeMRI(geometry=[f[SpaceTimeMRI.group_name][coord_name][()] \
        #                                   for coord_name in ["x_coordinates", "y_coordinates", "z_coordinates"]],
        #                         time=f[SpaceTimeMRI.group_name]["t_coordinates"][()],
        #                         vector_feature=f[SpaceTimeMRI.group_name]["vector_feature"][()].transpose((2,1,0,3,4)))

        lock = LockedFileH5Reader(path)
        lock.open()
        f = lock.file()
        # here comes the actual deserialization code
        mri = SpaceTimeMRI(geometry=[f[SpaceTimeMRI.group_name][coord_name][()] \
                                      for coord_name in ["x_coordinates", "y_coordinates", "z_coordinates"]],
                            time=f[SpaceTimeMRI.group_name]["t_coordinates"][()],
                            vector_feature=f[SpaceTimeMRI.group_name]["vector_feature"][()].transpose((2,1,0,3,4)))
        lock.close()
        return mri


class FlowMRI:
    """MRI datatype for time-dependent intensity & velocity measurements in 3d including axis information
    
    instance fields:
    geometry         ([np.ndarray(shape=(x_dim,), dtype=float), 
                       np.ndarray(shape=(y_dim,), dtype=float), 
                       np.ndarray(shape=(z_dim,), dtype=float)]):                     spatial coordinates along each axis
    time              (np.ndarray(shape=(t_dim,), dtype=float)):                      time coordinates
    intensity         (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim), dtype=float)):     intensity values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    velocity_mean     (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim,3), dtype=float)):   mean velocity values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    velocity_cov      (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim,3,3), dtype=float)): velocity covariance values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    
    class-level fields:
    group_name :  string                                                              path of datasets in HDF5-file
    """

    group_name = "flow-mri"
    
    def __init__(self, geometry: List[np.ndarray], time: np.ndarray, time_heart_cycle_period: float, intensity: np.ndarray, velocity_mean: np.ndarray, velocity_cov: np.ndarray):
        """Voxel-based parameters must be specified in (x,y,z,t,i)-order, Fortran will treat it in (i,t,x,y,z)-order.
           The index i is used as the component index (i.e. between 0..2 for mean and 0..5 for covariance of velocity field)
        """
        validate_spacetime_coordinates(geometry, time)
        validate_spacetime_scalar_feature(FlowMRI, geometry, time, intensity)
        validate_spacetime_vector_feature(FlowMRI, geometry, time, velocity_mean)
        validate_spacetime_matrix_feature(FlowMRI, geometry, time, velocity_cov)

        self.geometry = geometry
        self.time = time
        self.time_heart_cycle_period = time_heart_cycle_period
        self.intensity= intensity
        self.velocity_mean = velocity_mean
        self.velocity_cov= velocity_cov

    def write_hdf5(self, path: str):
        """Write this MRI to hpc-predict-io HDF5-format at path"""
        # file handling
        if os.path.exists(path):
            raise FileExistsError("Tried to open file %s, which exists already" % path)
    
        # with h5py.File(path, "w") as f:
        #     # here comes the actual serialization code (transposition to use Fortran memory layout)
        #     grp = f.create_group(FlowMRI.group_name)
        #     write_group_attribute(grp, "t_heart_cycle_period", self.time_heart_cycle_period)
        #     write_space_time_coordinates(grp, self.geometry, self.time)
        #     write_space_time_voxel_scalar_feature(grp, "intensity", self.intensity)
        #     write_space_time_voxel_vector_feature(grp, "velocity_mean", self.velocity_mean)
        #     write_space_time_voxel_matrix_feature(grp, "velocity_cov", self.velocity_cov)

        lock = LockedFileExclusiveH5Writer(path)
        lock.open()
        f = lock.file()
        # here comes the actual serialization code (transposition to use Fortran memory layout)
        grp = f.create_group(FlowMRI.group_name)
        write_group_attribute(grp, "t_heart_cycle_period", self.time_heart_cycle_period)
        write_space_time_coordinates(grp, self.geometry, self.time)
        write_space_time_voxel_scalar_feature(grp, "intensity", self.intensity)
        write_space_time_voxel_vector_feature(grp, "velocity_mean", self.velocity_mean)
        write_space_time_voxel_matrix_feature(grp, "velocity_cov", self.velocity_cov)
        lock.close()


    def read_hdf5(path: str):
        """Read MRI from file at path as hpc-predict-io HDF5-format"""
        # with h5py.File(path, "r") as f:
        #     # here comes the actual deserialization code
        #     if "t_heart_cycle_period" not in f[FlowMRI.group_name].attrs:
        #         logging.warning("Reading a FlowMRI with t_heart_cycle_period not set - using None in Python object.")
        #         time_heart_cycle_period = None
        #     else:
        #         time_heart_cycle_period = f[FlowMRI.group_name].attrs["t_heart_cycle_period"]
        #     return FlowMRI(geometry=[f[FlowMRI.group_name][coord_name][()] \
        #                                   for coord_name in ["x_coordinates", "y_coordinates", "z_coordinates"]],
        #                         time=f[FlowMRI.group_name]["t_coordinates"][()],
        #                         time_heart_cycle_period=time_heart_cycle_period,
        #                         intensity=f[FlowMRI.group_name]["intensity"][()].transpose((2,1,0,3)),
        #                         velocity_mean=f[FlowMRI.group_name]["velocity_mean"][()].transpose((2,1,0,3,4)),
        #                         velocity_cov=f[FlowMRI.group_name]["velocity_cov"][()].transpose((2,1,0,3,5,4)))

        lock = LockedFileH5Reader(path)
        lock.open()
        f = lock.file()
        # here comes the actual deserialization code
        if "t_heart_cycle_period" not in f[FlowMRI.group_name].attrs:
            logging.warning("Reading a FlowMRI with t_heart_cycle_period not set - using None in Python object.")
            time_heart_cycle_period = None
        else:
            time_heart_cycle_period = f[FlowMRI.group_name].attrs["t_heart_cycle_period"]
        mri = FlowMRI(geometry=[f[FlowMRI.group_name][coord_name][()] \
                                      for coord_name in ["x_coordinates", "y_coordinates", "z_coordinates"]],
                            time=f[FlowMRI.group_name]["t_coordinates"][()],
                            time_heart_cycle_period=time_heart_cycle_period,
                            intensity=f[FlowMRI.group_name]["intensity"][()].transpose((2,1,0,3)),
                            velocity_mean=f[FlowMRI.group_name]["velocity_mean"][()].transpose((2,1,0,3,4)),
                            velocity_cov=f[FlowMRI.group_name]["velocity_cov"][()].transpose((2,1,0,3,5,4)))
        lock.close()
        return mri

#TODO: Refactor this into class hierarchy with FlowMRI
class SegmentedFlowMRI:
    """MRI datatype for time-dependent intensity & velocity measurements in 3d including axis and segmentation information
    
    instance fields:
    geometry         ([np.ndarray(shape=(x_dim,), dtype=float), 
                       np.ndarray(shape=(y_dim,), dtype=float), 
                       np.ndarray(shape=(z_dim,), dtype=float)]):                     spatial coordinates along each axis
    time              (np.ndarray(shape=(t_dim,), dtype=float)):                      time coordinates
    intensity         (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim), dtype=float)):     intensity values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    velocity_mean     (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim,3), dtype=float)):   mean velocity values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    velocity_cov      (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim,3,3), dtype=float)): velocity covariance values over 
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    segmentation_prob (np.ndarray(shape=(x_dim,y_dim,z_dim,t_dim), dtype=float)):     segmentation probability values over
                                                                                      x_dim x y_dim x z_dim x t_dim - dimensional grid
    
    class-level fields:
    group_name :  string                                                              path of datasets in HDF5-file
    """

    group_name = "segmented-flow-mri"
    
    def __init__(self, geometry: List[np.ndarray], time: np.ndarray, time_heart_cycle_period: float, intensity: np.ndarray, velocity_mean: np.ndarray, velocity_cov: np.ndarray, segmentation_prob: np.ndarray):
        """Voxel-based parameters must be specified in (x,y,z,t,i)-order, Fortran will treat it in (i,t,x,y,z)-order.
           The index i is used as the component index (i.e. between 0..2 for mean and 0..5 for covariance of velocity field)
        """
        validate_spacetime_coordinates(geometry, time)
        validate_spacetime_scalar_feature(SegmentedFlowMRI, geometry, time, intensity)
        validate_spacetime_vector_feature(SegmentedFlowMRI, geometry, time, velocity_mean)
        validate_spacetime_matrix_feature(SegmentedFlowMRI, geometry, time, velocity_cov)
        validate_spacetime_scalar_feature(SegmentedFlowMRI, geometry, time, segmentation_prob)

        self.geometry = geometry
        self.time = time
        self.time_heart_cycle_period = time_heart_cycle_period
        self.intensity= intensity
        self.velocity_mean = velocity_mean
        self.velocity_cov= velocity_cov
        self.segmentation_prob = segmentation_prob


    def write_hdf5(self, path: str):
        """Write this MRI to hpc-predict-io HDF5-format at path"""
        # file handling
        if os.path.exists(path):
            raise FileExistsError("Tried to open file %s, which exists already" % path)
    
        # with h5py.File(path, "w") as f:
        #     # here comes the actual serialization code (transposition to use Fortran memory layout)
        #     grp = f.create_group(SegmentedFlowMRI.group_name)
        #     write_group_attribute(grp, "t_heart_cycle_period", self.time_heart_cycle_period)
        #     write_space_time_coordinates(grp, self.geometry, self.time)
        #     write_space_time_voxel_scalar_feature(grp, "intensity", self.intensity)
        #     write_space_time_voxel_vector_feature(grp, "velocity_mean", self.velocity_mean)
        #     write_space_time_voxel_matrix_feature(grp, "velocity_cov", self.velocity_cov)
        #     write_space_time_voxel_scalar_feature(grp, "segmentation_prob", self.segmentation_prob)

        lock = LockedFileExclusiveH5Writer(path)
        lock.open()
        f = lock.file()
        # here comes the actual serialization code (transposition to use Fortran memory layout)
        grp = f.create_group(SegmentedFlowMRI.group_name)
        write_group_attribute(grp, "t_heart_cycle_period", self.time_heart_cycle_period)
        write_space_time_coordinates(grp, self.geometry, self.time)
        write_space_time_voxel_scalar_feature(grp, "intensity", self.intensity)
        write_space_time_voxel_vector_feature(grp, "velocity_mean", self.velocity_mean)
        write_space_time_voxel_matrix_feature(grp, "velocity_cov", self.velocity_cov)
        write_space_time_voxel_scalar_feature(grp, "segmentation_prob", self.segmentation_prob)
        lock.close()


    def read_hdf5(path: str):
        """Read MRI from file at path as hpc-predict-io HDF5-format"""
        # with h5py.File(path, "r") as f:
        #     # here comes the actual deserialization code
        #     if "t_heart_cycle_period" not in f[SegmentedFlowMRI.group_name].attrs:
        #         logging.warning("Reading a SegmentedFlowMRI with t_heart_cycle_period not set - using None in Python object.")
        #         time_heart_cycle_period = None
        #     else:
        #         time_heart_cycle_period = f[SegmentedFlowMRI.group_name].attrs["t_heart_cycle_period"]
        #     return SegmentedFlowMRI(geometry=[f[SegmentedFlowMRI.group_name][coord_name][()] \
        #                                   for coord_name in ["x_coordinates", "y_coordinates", "z_coordinates"]],
        #                         time=f[SegmentedFlowMRI.group_name]["t_coordinates"][()],
        #                         time_heart_cycle_period=time_heart_cycle_period,
        #                         intensity=f[SegmentedFlowMRI.group_name]["intensity"][()].transpose((2,1,0,3)),
        #                         velocity_mean=f[SegmentedFlowMRI.group_name]["velocity_mean"][()].transpose((2,1,0,3,4)),
        #                         velocity_cov=f[SegmentedFlowMRI.group_name]["velocity_cov"][()].transpose((2,1,0,3,5,4)),
        #                         segmentation_prob=f[SegmentedFlowMRI.group_name]["segmentation_prob"][()].transpose((2,1,0,3)))

        lock = LockedFileH5Reader(path)
        lock.open()
        f = lock.file()
        # here comes the actual deserialization code
        if "t_heart_cycle_period" not in f[SegmentedFlowMRI.group_name].attrs:
            logging.warning("Reading a SegmentedFlowMRI with t_heart_cycle_period not set - using None in Python object.")
            time_heart_cycle_period = None
        else:
            time_heart_cycle_period = f[SegmentedFlowMRI.group_name].attrs["t_heart_cycle_period"]
        mri = SegmentedFlowMRI(geometry=[f[SegmentedFlowMRI.group_name][coord_name][()] \
                                      for coord_name in ["x_coordinates", "y_coordinates", "z_coordinates"]],
                            time=f[SegmentedFlowMRI.group_name]["t_coordinates"][()],
                            time_heart_cycle_period=time_heart_cycle_period,
                            intensity=f[SegmentedFlowMRI.group_name]["intensity"][()].transpose((2,1,0,3)),
                            velocity_mean=f[SegmentedFlowMRI.group_name]["velocity_mean"][()].transpose((2,1,0,3,4)),
                            velocity_cov=f[SegmentedFlowMRI.group_name]["velocity_cov"][()].transpose((2,1,0,3,5,4)),
                            segmentation_prob=f[SegmentedFlowMRI.group_name]["segmentation_prob"][()].transpose((2,1,0,3)))
        lock.close()
        return mri

def read_hdf5(path: str):
    """Read MRI from file at path as hpc-predict-io HDF5-format"""
    mr_io_classes = { cls.group_name : cls.read_hdf5 for cls in [SegmentedFlowMRI, FlowMRI, SpaceTimeMRI, SpatialMRI] }
    mr_io_reader = None

    lock = LockedFileH5Reader(path)
    lock.open()
    f = lock.file()
    for group_name in f.keys():
        if mr_io_classes.get(group_name) is not None:
            mr_io_reader = mr_io_classes[group_name]
            break
    lock.close()
    return mr_io_reader(path)

