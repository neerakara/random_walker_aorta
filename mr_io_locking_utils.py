import os
import fcntl
import time


class LockedFileExclusiveWriter:
    def __init__(self, filename):
        self._filename = filename

    def open(self):
        self._f = open(self._filename,'w')
        fcntl.flock(self._f, fcntl.LOCK_EX)

    def file(self):
        if not hasattr(self, "_f") or self._f is None:
            raise RuntimeError("File object not defined - first call open().")
        return self._f

    def close(self):
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close() # fcntl.flock(self._f, fcntl.LOCK_UN) not needed
        self._f = None


class LockedFileReader:
    def __init__(self, filename):
        self._filename = filename

    def open(self):
        while not os.path.exists(self._filename):
            time.sleep(1)

        if os.path.exists(self._filename):
            self._f = open(self._filename, 'r')

            # Ensure writer before reader access
            while os.lseek(self._f.fileno(), 0, os.SEEK_END) == 0:
                pass
            os.lseek(self._f.fileno(), 0, os.SEEK_SET)

            fcntl.flock(self._f, fcntl.LOCK_SH)

    def file(self):
        if not hasattr(self, "_f") or self._f is None:
            raise RuntimeError("File object not defined - first call open().")
        return self._f

    def close(self):
        #fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close() # fcntl.flock(self._f, fcntl.LOCK_UN) not needed
        self._f = None


import h5py


class LockedFileExclusiveH5Writer:
    def __init__(self, filename):
        self._filename = filename

    def open(self):
        self._f = h5py.File(self._filename, 'w') # makes fcntl.flock(self._f.id.get_vfd_handle(), fcntl.LOCK_EX) redundant

    def file(self):
        if not hasattr(self, "_f") or self._f is None:
            raise RuntimeError("h5py.File object not defined - first call open().")
        return self._f

    def close(self):
        self._f.flush()
        os.fsync(self._f.id.get_vfd_handle())
        self._f.close() #fcntl.flock(self._f.id.get_vfd_handle(), fcntl.LOCK_UN) not needed
        self._f = None


class LockedFileH5Reader:
    def __init__(self, filename):
        self._filename = filename

    def open(self):
        while not os.path.exists(self._filename):
            time.sleep(1)

        if os.path.exists(self._filename):
            self._f_tmp = open(self._filename, 'r') # extra file descriptor necessary unfortunately as h5py.File provides only non-blocking lock acquisition

            # Ensure writer before reader access
            while os.lseek(self._f_tmp.fileno(), 0, os.SEEK_END) == 0:
                pass
            os.lseek(self._f_tmp.fileno(), 0, os.SEEK_SET)

            fcntl.flock(self._f_tmp, fcntl.LOCK_SH)

            self._f = h5py.File(self._filename, 'r')

            self._f_tmp.close() # makes fcntl.flock(self._f_tmp, fcntl.LOCK_UN) redundant

    def file(self):
        if not hasattr(self, "_f") or self._f is None:
            raise RuntimeError("File object not defined - first call open().")
        return self._f

    def close(self):
        self._f.close() # fcntl.flock(self._f.id.get_vfd_handle(), fcntl.LOCK_UN) not needed
        self._f = None

