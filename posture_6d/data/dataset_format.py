# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_keys
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from open3d import geometry, utility, io
import os
import glob
import shutil
import pickle
import cv2
import time
from tqdm import tqdm
import warnings

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable

from . import Posture, JsonIO, JSONDecodeError, extract_doc
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

FT = TypeVar('FT', bound='DatasetFormat')
DCT = TypeVar('DCT')
DST = TypeVar('DST')
from numpy import ndarray

def as_dict(ids, objs):
    if objs is None:
        return None
    else:
        return dict(zip(ids, objs))

def savetxt_func(fmt=...):
    return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

def loadtxt_func(shape:tuple[int]):
    return lambda path: np.loadtxt(path).reshape(shape)

class WriteController():
    '''
    control the write operation.
    the subclass of WriteController must implement :
    * start_writing
    * stop_writing
    '''
    class __Writer():
        def __init__(self, writecontroller:"WriteController") -> None:
            self.writecontroller = writecontroller
            self.__overwrite_allowed = False

        def allow_overwriting(self):
            self.__overwrite_allowed = True
            return self


        def __enter__(self):
            self.writecontroller.start_writing(self.__overwrite_allowed)
            self.writecontroller.is_writing = True
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                self.writecontroller.stop_writing()
                self.writecontroller.is_writing = False
                return True
    
    def __init__(self) -> None:
        self.__writer = self.__Writer(self)
        self.__is_writing = False

    @property
    def is_writing(self):
        return self.__is_writing

    @is_writing.setter
    def is_writing(self, value:bool):
        self.__is_writing = bool(value)

    @property
    def writer(self):
        return self.__writer

    def start_writing(self, overwrite_allowed = False):
        pass

    def stop_writing(self):
        pass

#### Warning Types ####
class ClusterDataIOError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterParaWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterIONotExecutedWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterNotRecommendWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

#### BASE #####
class _DataCluster(ABC, Generic[FT, DCT], WriteController):
    '''
    This is a private class representing a data cluster used for managing datasets with a specific format.

    # attr
    ----
    * self.format_obj: DatasetFormat
    * self.closed: bool, Control the shielding of reading and writing, 
        if it is true, the instance will not write, and the read will get None
    * register: bool, whether to register to format_obj
    * _incomplete: bool, whether the data is incomplete
    * _closed: bool, Indicates whether the cluster is closed or open.
    * _readonly: bool, Indicates whether the cluster is read-only or write-enabled.
    * changes_unsaved: bool, Indicates if any changes have been made to the cluster.
    * directory: str, Directory path for the cluster.

    # property
    -----
    * overwrite_allowed: bool, Control the shielding of writing,
    * cluster_data_num: int, the number of data in the cluster
    * cluster_data_i_upper: int, the upper of the iterator, it is the max index of the iterator + 1
    * changed_since_opening: bool, Indicates whether the cluster has been modified since last opening.

    # method
    -----
    abstract method:
    -----
    - __len__: return the number of data in the cluster
    - keys: return the keys of the cluster
    - values: return the values of the cluster
    - items: return the items of the cluster(key and value)
    - _read: read data from the cluster
    - _write: write data to the cluster
    - _clear: clear all data of the cluster

    recommend to implement:
    -----
    - _init_attr: initialize additional attributes specified by subclasses.
    - _update_cluster_inc: update the incremental modification of the cluster after writing
    - _update_cluster_all: update the state of the cluster after writing
    - __getitem__: return the value of the key
    - __setitem__: set the value of the key    

    not need to implement:
    -----
    - __iter__: return the iterator of the cluster
    - open: open the cluster for operation.
    - close: close the cluster, preventing further operations.
    - is_close: check if the cluster is closed.
    - set_readonly: set the cluster as read-only or write-enabled.
    - is_readonly: check if the cluster is read-only.
    - _read_decorator: decorator function to handle reading operations when the cluster is closed.
    - _write_decorator: decorator function to handle writing operations when the cluster is closed or read-only.
    - clear: clear any data in the cluster. Subclasses may implement _clear.
    - read: read data from the cluster. Subclasses must implement _read.
    - write: write data to the cluster. Subclasses must implement _write.
    '''
    def __init__(self, format_obj:FT, sub_dir: str, register=True, *args, **kwargs) -> None:
        '''
        Initialize the data cluster with the provided format_obj, sub_dir, and registration flag.
        '''
        WriteController.__init__(self)    
        self.format_obj:FT = format_obj
        self.sub_dir = sub_dir
        self.register = register
        self._incomplete = self.format_obj.incomplete
        self._error_to_load = False
        self.__closed = True  # Indicates whether the cluster is closed or open.
        self.__readonly = True  # Indicates whether the cluster is read-only or write-enabled.
        self.__overwrite_allowed = False
        self._changed_since_opening = False  # Indicates if any changes have been made to the cluster.
        self.directory = os.path.join(format_obj.directory, self.sub_dir)  # Directory path for the cluster.
        self._data_i_upper = 0  

        self.__cluster_read_func:Callable =  self._read_decorator(self.__class__._read)
        self.__cluster_write_func:Callable = self._write_decorator(self.__class__._write)
        self.__cluster_clear_func:Callable = self._write_decorator(self.__class__._clear) 

        self._init_attr(*args, **kwargs)  # Initializes additional attributes specified by subclasses.
     
        if os.path.exists(self.directory) and not self._error_to_load:
            self.open()  # Opens the cluster for operation.
        else:
            self.close()
        
        self.register_to_format()

    @property
    def overwrite_allowed(self):
        '''Property that returns whether the cluster format allows write operations.'''
        return self.__overwrite_allowed

    @property
    def cluster_data_num(self):
        return len(self)

    @property
    def cluster_data_i_upper(self):
        return max(self.keys()) + 1 if len(self) > 0 else 0

    @property
    def changed_since_opening(self):
        '''Indicates whether the cluster has been modified since last opening.'''
        return self._changed_since_opening
    
    @changed_since_opening.setter
    def changed_since_opening(self, value:bool):
        self._changed_since_opening = bool(value)
        self.format_obj.updated = True

    @abstractmethod
    def __len__(self):
        '''Returns the number of data in the cluster.'''
        pass     

    @abstractmethod
    def keys(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def values(self) -> Iterable[DCT]:
        pass

    @abstractmethod
    def items(self) -> Iterable[tuple[Any, DCT]]:
        pass

    @abstractmethod
    def _read(self, data_i, *arg, **kwargs) -> DCT:
        pass

    @abstractmethod
    def _write(self, data_i, value:DCT, *arg, **kwargs):
        pass

    @abstractmethod
    def _clear(self, *arg, **kwargs):
        pass

    @abstractmethod
    def _copyto(self, dst: str, *args, **kwargs):
        pass

    def _init_attr(self, *args, **kwargs):
        '''Method to initialize additional attributes specified by subclasses.'''
        pass

    def _update_cluster_inc(self, data_i, *args, **kwargs):
        '''
        update the state of the cluster after writing
        '''
        pass

    def _update_cluster_all(self, *args, **kwargs):
        pass
   
    def __getitem__(self, data_i) -> DCT:
        return self.read(data_i)
    
    def __setitem__(self, data_i, value:DCT):
        return self.write(data_i, value)

    def check_key(self, key) -> bool:
        return True

    def __iter__(self) -> Iterable[DCT]:
        return self.values()

    def _open(self):
        '''Method to open the cluster for operation.'''
        self.__closed = False  # Marks the cluster as open for operation.        
        if self._incomplete:
            readonly = self.__readonly  # Stores the read-only flag.
            self.set_readonly(False)  # Sets the cluster as write-enabled.
            self.clear(ignore_warning=True)  # Clears any incomplete data if present.
            self._incomplete = False
            self.set_readonly(readonly)  # Restores the read-only flag.

    def _close(self):
        self._changed_since_opening = False  # Resets the updated flag to false.
        self.__closed = True  # Marks the cluster as closed.

    def open(self):
        if self.__closed == True:
            self._open()
            return True
        return False

    def close(self):
        '''Method to close the cluster, preventing further operations.'''
        if self.__closed == False:
            self._close()
            return True
        return False

    def is_close(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self.__closed:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self.__closed

    def is_open(self):
        return not self.is_close()

    def set_readonly(self, readonly=True):
        '''Method to set the cluster as read-only or write-enabled.'''
        self.__readonly = readonly

    def set_writable(self, writable=True):
        '''Method to set the cluster as writable or read-only.'''
        self.set_readonly(not writable)

    def set_overwrite_allowed(self, overwrite_allowed=True):
        '''Method to set the cluster as writable or read-only.'''
        self.__overwrite_allowed = overwrite_allowed

    def is_readonly(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self.__readonly:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self.__readonly
    
    def is_writable(self):
        return not self.is_readonly()

    def is_overwrite_allowed(self):
        return self.__overwrite_allowed

    @staticmethod
    def _read_decorator(func):
        '''
        brief
        -----
        Decorator function to handle reading operations when the cluster is closed. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        def wrapper(self: "_DataCluster", data_i, *args, force = False, **kwargs):
            if not force:
                if self.is_close(with_warning=True):
                    return None
            try:
                rlt = func(self, data_i, *args, **kwargs)  # Calls the original function.
            except ClusterDataIOError:
                rlt = None
            return rlt
        return wrapper

    @staticmethod
    def _write_decorator(func):
        '''
        brief
        -----
        Decorator function to handle writing operations when the cluster is closed or read-only. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        if the cluster is read-only and the decorated function is a writing operation, the decorated function will not be executed and return None.\n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        def wrapper(self: "_DataCluster", data_i = None, value = None, *args, force = False, **kwargs):
            if force:
                orig_closed = self.__closed
                orig_readonly = self.__readonly
                self.open()
                self.set_writable()
            if self.is_close(with_warning=True) or self.is_readonly(with_warning=True):
                return None
            elif not self.overwrite_allowed and data_i in self.keys() and not force:
                warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                            is not allowed to overwitre, any write operation will not be executed.",
                                ClusterIONotExecutedWarning)
                return None
            try:
                rlt = func(self, data_i, value, *args, **kwargs)  # Calls the original function.
            except ClusterDataIOError:
                rlt = None
            else:
                self.changed_since_opening = True  # Marks the cluster as updated after writing operations.
                if data_i is None:
                    self._update_cluster_all(*args, **kwargs)
                else:
                    self._update_cluster_inc(data_i, *args, **kwargs)

            if force:
                if orig_closed:
                    self.close()
                if orig_readonly:
                    self.set_readonly()
            return rlt
        return wrapper

    def clear(self, *, force = False, ignore_warning = False):
        '''
        Method to clear any data in the cluster. Subclasses may implement this method.
        * it is dargerous
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            self.__cluster_clear_func(self, force = force)
            return True
        else:
            return False

    def read(self, data_i, *args, force = False, **kwargs)->DCT:
        '''
        Method to read data from the cluster. Subclasses must implement this method.
        '''
        return self.__cluster_read_func(self, data_i, *args, force=force, **kwargs)

    def write(self, data_i, value:DCT, *args, force = False, **kwargs):
        '''
        Method to write data to the cluster. Subclasses must implement this method.

        parameter
        ----
        * data_i: Any, the key of the data
        * value: DCT, the value of the data
        * force: bool, whether to force write, it's not recommended to use force=True if you want to write a lot of data
        '''
        assert self.check_key(data_i)
        return self.__cluster_write_func(self, data_i, value, *args, force = force, **kwargs)

    def copyto(self, dst: str, cover:bool, *args, **kwargs):
        '''
        copy the cluster to dst
        '''
        if os.path.exists(dst):
            if cover:
                warnings.warn(f"{dst} exists, it will be deleted", ClusterParaWarning)
                shutil.rmtree(dst)
            else:
                raise ValueError(f"{dst} exists, it will not be deleted, please set cover=True")
        self._copyto(dst, *args, **kwargs)

    def start_writing(self, overwrite_allowed = False):
        '''
        rewrite the method of WriteController
        '''
        self.open()
        self.set_writable()
        self.set_overwrite_allowed(overwrite_allowed)

    def stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        self.set_readonly()
        self.close()
        self.set_overwrite_allowed(False)
        self._update_cluster_all()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.format_obj.__class__.__name__}, {self.sub_dir}) at {hex(id(self))}"

    def __del__(self):
        '''
        save _data_i_dir_map and _data_i_appendnames
        '''
        self.close()

    def register_to_format(self):
        if self.register:
            self.format_obj.cluster_map[self.directory] = self

class JsonDict(_DataCluster[FT, DCT], dict):
    '''
    dict for base json
    ----
    it is a subclass of dict, so it can be used as a dict \n
    returns None if accessing an key that does not exist

    attr
    ----
    see _DataCluster

    method
    ----
    see _DataCluster
    * clear: clear all data of the dict and clear the json file
    '''
    SAVE_IMMIDIATELY = 0
    SAVE_AFTER_CLOSE = 1
    SAVE_STREAMLY = 2

    class _Placeholder():
        def __init__(self) -> None:
            pass

    def __init__(self, format_obj:FT, sub_dir, register = True, *args, **kwargs):       
        dict.__init__(self)
        super().__init__(format_obj, sub_dir, register, *args, **kwargs)

    @property
    def save_mode(self):
        return self.__save_mode
    
    @save_mode.setter
    def save_mode(self, mode):
        assert mode in [self.SAVE_IMMIDIATELY, self.SAVE_AFTER_CLOSE, self.SAVE_STREAMLY]
        if self.is_writing and mode != self.SAVE_STREAMLY:
            warnings.warn("can't change save_mode from SAVE_STREAMLY to the others while writing streamly", 
                          ClusterParaWarning)
        if self.__save_mode == self.SAVE_AFTER_CLOSE and mode != self.SAVE_AFTER_CLOSE and self.is_open():
            self.save()
        self.__save_mode = mode

    @property
    def write_streamly(self):
        return self.save_mode == self.SAVE_STREAMLY

    def __len__(self) -> int:
        super().__len__()
        return dict.__len__(self)

    def keys(self):
        super().keys()
        return dict.keys(self)

    def values(self):
        super().values()
        return dict.values(self)
    
    def items(self):
        super().items()
        return dict.items(self)
    
    def _read(self, data_i, *arg, **kwargs):
        super()._read(data_i, *arg, **kwargs)
        try:
            return dict.__getitem__(self, data_i)
        except KeyError:
            raise ClusterDataIOError(f"key {data_i} does not exist")

    def _write(self, data_i, value, *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)

        if self.save_mode == self.SAVE_IMMIDIATELY:
            rlt = dict.__setitem__(self, data_i, value)
            self.save()
        elif self.save_mode == self.SAVE_AFTER_CLOSE:
            rlt = dict.__setitem__(self, data_i, value)
        elif self.save_mode == self.SAVE_STREAMLY:
            set_value = self._Placeholder()
            if not self.is_writing:
                self.start_writing() # auto start writing if save_mode is SAVE_STREAMLY
            self.stream.write({data_i: value})
            rlt = dict.__setitem__(self, data_i, set_value)
        return rlt

    def _clear(self, *arg, **kwargs):
        with open(self.directory, 'w'):
            pass
        dict.clear(self)

    def _copyto(self, dst: str, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        os.makedirs(os.path.split(dst)[0], exist_ok=True)
        shutil.copy(self.directory, dst)

    def _init_attr(self, *args, **kwargs):
        _DataCluster._init_attr(self, *args, **kwargs)
        self.reload()
        self.__save_mode = self.SAVE_AFTER_CLOSE
        self.stream = JsonIO.Stream(self.directory)

    def __iter__(self) -> Iterator:
        return _DataCluster.__iter__(self)

    def _close(self):
        '''
        rewrite the method of WriteController
        '''
        if self.is_writing:
            self.stop_writing()
        if self.save_mode == self.SAVE_AFTER_CLOSE:
            self.sort()
            self.save()
        super()._close()

    def update(self, _m:Union[dict, Iterable[tuple]] = None, **kw):
        if _m is None:
            warnings.warn(f"It's not recommended to use update() for {self.__class__.__name__} \
                for it will call {self._update_cluster_all.__name__} and spend more time. \
                nothine is done. \
                use {self.__setitem__.__name__} or {self.write.__name__} or {self.__setitem__.__name__} instead", ClusterNotRecommendWarning)
        elif isinstance(_m, dict):
            for k, v in _m.items():
                assert self.checkkey(k), f"key {k} is not allowed"
                self[k] = v
        elif isinstance(_m, Iterable):
            assert all([len(x) == 2 for x in _m])
            for k, v in _m:
                assert self.checkkey(k), f"key {k} is not allowed"
                self[k] = v
        else:
            raise TypeError(f"unsupported type: {type(_m)}")

    def reload(self, value = {}):
        dict.clear(self)
        if isinstance(value, dict) and len(value) > 0:
            pass
        elif os.path.exists(self.directory):
            try:
                value = JsonIO.load_json(self.directory)
            except JSONDecodeError:
                self._error_to_load = True
                value = {}
        else:
            value = {}
        dict.update(self, value)

    @_DataCluster._write_decorator
    def sort(self, *arg, **kwargs):
        '''
        sort by keys
        '''
        if self.is_writing:
            warnings.warn(f"It's no effect to call {self.sort.__name__} while writing", ClusterNotRecommendWarning)
            return 
        new_dict = dict(sorted(self.items(), key=lambda x: x[0]))
        self.reload(new_dict)
        if self.save_mode == self.SAVE_IMMIDIATELY:
            self.save()

    def start_writing(self, overwrite_allowed = False):
        '''
        rewrite the method of WriteController
        '''
        _DataCluster.start_writing(self, overwrite_allowed)
        self.save_mode = self.SAVE_STREAMLY
        self.stream.open()

    def stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        _DataCluster.stop_writing(self)
        self.stream.close()
        self.reload()

    def save(self):
        JsonIO.dump_json(self.directory, self)

class Elements(_DataCluster[FT, DCT]):
    '''
    elements manager
    ----
    Returns None if accessing an data_id that does not exist \n
    will not write if the element is None \n 
    it can be used as an iterator, the iterator will return (data_id, element) \n

    attr
    ----
    see _DataCluster
    * readfunc:  Callable, how to read one element from disk, the parameter must be (path)
    * writefunc: Callable, how to write one element to disk, the parameter must be (path, element)
    * suffix: str, the suffix of the file
    * filllen: int, to generate the store path
    * fillchar: str, to generate the store path
    * _data_i_dir_map: dict[int, str], the map of data_id and directory name
    * data_i_upper: int, the max index of the iterator

    method
    ----
    * __len__: return the number of elements
    * __iter__: return the iterator
    * __next__: return the next element
    * read: read one element from disk with the logic of self.readfunc
    * write: write one element to disk with the logic of self.writefunc
    * format_path: format the path of the element
    * clear: clear all data of the dict and clear directory


    '''
    def __init__(self, 
                format_obj:FT,
                sub_dir,
                register = True,
                read_func:Callable = lambda x: None, 
                write_func:Callable = lambda x,y: None, 
                suffix:str = '.txt', 
                filllen = 6, 
                fillchar = '0', *,
                alternative_suffix:list[str] = []) -> None:
        super().__init__(format_obj, sub_dir, register, read_func, write_func, suffix, filllen, fillchar, alternative_suffix = alternative_suffix)

    @property
    def data_i_dir_map(self):
        if len(self._data_i_dir_map) == 0:
            self._update_cluster_all()
        return self._data_i_dir_map
    
    @property
    def data_i_appendnames(self):
        if len(self._data_i_appendnames) == 0:
            self._update_cluster_all()
        return self._data_i_appendnames

    @property
    def cache_mode(self):
        return self.__cache_mode
    
    @cache_mode.setter
    def cache_mode(self, value:bool):
        if self.load_cache():
            self.__cache_mode = bool(value)
        else:
            warnings.warn("no cache file, can not set to  cache mode. call save_cache' first", ClusterParaWarning)

    def __len__(self):
        '''
        Count the total number of files in the directory
        '''
        super().__len__()
        # count = 0
        # for root, dirs, files in os.walk(self.directory):
        #     count += len(glob.glob(os.path.join(root, f'*{self.suffix}')))
        return len(self.data_i_dir_map)

    def keys(self):
        '''
        brief
        -----
        return a generator of data_i
        * Elements is not a dict, so it can't be used as a dict.
        '''
        super().keys()
        def data_i_generator():
            for i in self.data_i_dir_map.keys():
                yield i
        return data_i_generator()
    
    def values(self):
        super().values()
        def value_generator():
            for i in self.data_i_dir_map.keys():
                yield self.read(i)
        return value_generator()
    
    def items(self):
        super().items()
        def items_generator():
            for i in self.data_i_dir_map.keys():
                yield i, self.read(i)
        return items_generator()

    def _read(self, data_i, appdir = "", appname = "", *arg, **kwargs)->DCT:
        super()._read(data_i, *arg, **kwargs)
        if not self.cache_mode:
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            if not os.path.exists(path):
                path = self.auto_path(data_i)
            if os.path.exists(path):
                return self.read_func(path, *arg, **kwargs)
            else:
                raise ClusterDataIOError(f"can't find {path}")
        else:
            if data_i in self.__cache:
                return self.__cache[data_i]
            else:
                raise ClusterDataIOError(f"can't find {path}")
        
    def _write(self, data_i, value:DCT, appdir = "", appname = "", *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)
        if not self.cache_mode:
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            dir_ = os.path.split(path)[0]
            os.makedirs(dir_, exist_ok=True)
            if value is not None:
                self.write_func(path, value, *arg, **kwargs)
        else:
            warnings.warn("can't write in cache_mode", ClusterIONotExecutedWarning)

    def _clear(self, *arg, **kwargs):
        super()._clear(*arg, **kwargs)
        shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def _copyto(self, dst: str, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        if not self.cache_mode:
            shutil.copytree(self.directory, dst)
        else:
            os.makedirs(dst, exist_ok=True)
            shutil.copy(os.path.join(self.directory, self.__data_i_dir_map_name), 
                        os.path.join(dst, self.__data_i_dir_map_name))
            shutil.copy(os.path.join(self.directory, self.__data_i_appendnames_name),
                        os.path.join(dst, self.__data_i_appendnames_name))
            shutil.copy(os.path.join(self.directory, self.__cache_name),
                        os.path.join(dst, self.__cache_name))

    def _init_attr(self, read_func, write_func, suffix, filllen, fillchar, alternative_suffix = [], *args, **kwargs):
        def _check_dot(suffix:str):
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            return suffix

        super()._init_attr(*args, **kwargs)
        assert isinstance(suffix, str), f"suffix must be str, not {type(suffix)}"
        assert isinstance(filllen, int), f"filllen must be int, not {type(filllen)}"
        assert isinstance(fillchar, str), f"fillchar must be str, not {type(fillchar)}"
        assert isinstance(alternative_suffix, (list, tuple)), f"alternative_suffix must be list or tuple, not {type(alternative_suffix)}"
        self.__data_i_dir_map_name = "data_i_dir_map.elmm"
        self.__data_i_appendnames_name = "data_i_appendnames.elmm"
        self.__cache_name = "cache.elmcache"

        self.filllen    = filllen
        self.fillchar   = fillchar
        self.suffix     = suffix
        self.__alternative_suffix = alternative_suffix
        self.suffix = _check_dot(self.suffix)
        self.__alternative_suffix = [_check_dot(x) for x in self.__alternative_suffix]

        self.read_func  = read_func
        self.write_func = write_func
        
        self._data_i_dir_map = {}
        self._data_i_appendnames = {}

        self.__cache_mode = False
        self.__cache = {}

        self.load_maps()
        if len(self) == 0:
            if self.load_cache():
                self.cache_mode = True

    def _update_cluster_inc(self, data_i, appdir = "", appname = "", *arg, **kwargs):
        self._data_i_dir_map[data_i] = appdir
        self._data_i_appendnames.setdefault(data_i, []).append(appname)

    def _update_cluster_all(self, *args, **kwargs):
        # if len(self) > 2000:
        #     print(f'init {self.directory} data_i_dir_map, this may take a while...')        
        self._data_i_dir_map.clear()
        self._data_i_appendnames.clear()
        paths = glob.glob(os.path.join(self.directory, "**/*" + self.suffix), recursive=True)
        for path in paths:
            data_i, appdir, appname = self.parse_path(path)
            self._update_cluster_inc(data_i, appdir=appdir, appname=appname)
        # sort by data_i
        self._data_i_dir_map = dict(sorted(self._data_i_dir_map.items(), key=lambda x:x[0]))
        self._data_i_appendnames = dict(sorted(self._data_i_appendnames.items(), key=lambda x:x[0]))

        if len(self._data_i_dir_map) > 0:
            self.save_maps()

    def __getitem__(self, idx):
        return self.read(data_i=idx)
        
    def __setitem__(self, idx, value):
        if idx in self.data_i_dir_map:
            appdir, appname = self.auto_path(idx, return_app=True)
            self.write(idx, value, appdir=appdir, appname = appname)
        else:
            raise KeyError(f'idx {idx} not in {self.directory}, if you want to add new data, \
                           please use method:write to specify the appdir and appname')

    def __iter__(self):
        self.data_i_dir_map
        return self.values()
    
    def _open(self):
        if not os.path.exists(self.directory):
            print(f"Elements: {self.directory} is new, it will be created")
            os.makedirs(self.directory, exist_ok=True)
        return super()._open()

    def _close(self):
        if self.changed_since_opening:
            # if not self.check_storage():
            self._update_cluster_all()     
            self.save_maps()
        return super()._close()

    def read(self, data_i, appdir = "", appname = "", *arg, force = False, **kwarg):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        return super().read(data_i, appdir = appdir, appname = appname, *arg, force = force, **kwarg)

    def write(self, data_i, element:DCT, appdir = "", appname = "", *arg, force = False, **kwarg):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * element: the element to be written
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        return super().write(data_i, element, appdir = appdir, appname = appname, *arg, force=force, **kwarg)

    def format_base_name(self, data_i):
        return "{}".format(str(data_i).rjust(self.filllen, "0"))

    def format_path(self, data_i, appdir = "", appname = "", **kw):
        '''
        format the path of data_i
        '''
        if appname and appname[0] != '_':
            appname = '_' + appname # add '_' before appname
        return os.path.join(self.directory, appdir, 
                            "{}{}{}".format(
                                self.format_base_name(data_i), 
                                appname, 
                                self.suffix))

    def parse_path(self, path:str):
        '''
        parse the path to get data_i, appdir, appname, it is the reverse operation of format_path
        '''
        appdir, file = os.path.split(os.path.relpath(path, self.directory))
        filename = os.path.splitext(file)[0]
        split_filename = filename.split('_')
        mainname = split_filename[0]
        try:
            appname  = "_".join(split_filename[1:])
        except IndexError:
            appname  = ""
        data_i = int(mainname)
        return data_i, appdir, appname

    def auto_path(self, data_i, return_app = False):
        '''
        auto find the path of data_i. \n
        * if data_i has multiple appendnames, raise IndexError
        '''
        if data_i in self.data_i_dir_map and data_i in self.data_i_appendnames:
            appdir = self.data_i_dir_map[data_i]
            appendnames = self.data_i_appendnames[data_i]
            if len(appendnames) == 1:
                appname = appendnames[0]
            else:
                raise ClusterDataIOError(f'idx {data_i} has more than one appendname: {appendnames}, its path is ambiguous. \
                                You must specify the appname by using method:read(data_i, appname=...)')
        else:
            appdir = ""
            appname = ""
        if not return_app:
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            if os.path.exists(path):
                return path
            else:
                raise ClusterDataIOError(f'idx {data_i} has no file in {self.directory}')
                return None
        else:
            return appdir, appname

    def check_storage(self):
        paths = glob.glob(os.path.join(self.directory, "**/*" + self.suffix), recursive=True)
        files = [int(os.path.split(path)[1][:self.filllen]) for path in paths]
        files_set = set(files)
        cur_keys_set = set(self.keys())        
        if len(files_set) != len(files):
            raise ValueError(f"there are duplicate files in {self.directory}")
        if len(files_set) != len(cur_keys_set):
            return False
        elif files_set == cur_keys_set:
            return True
        else:
            return False

    def save_maps(self):
        for name, map in zip((self.__data_i_dir_map_name, self.__data_i_appendnames_name), 
                             (self._data_i_dir_map, self._data_i_appendnames)):
            path = os.path.join(self.directory, os.path.splitext(name)[0] + ".npy")
            path_elmm = os.path.join(self.directory, name)
            np.save(path, map)
            try:
                os.remove(path_elmm)
            except FileNotFoundError:
                pass
            os.rename(path, path_elmm)

    def load_maps(self):
        data_i_dir_map_path         = os.path.join(self.directory, self.__data_i_dir_map_name)
        data_i_appendnames_path     = os.path.join(self.directory, self.__data_i_appendnames_name)
        if os.path.exists(data_i_dir_map_path) and os.path.exists(data_i_appendnames_path):
            self._data_i_dir_map        = np.load(data_i_dir_map_path, allow_pickle=True).item()
            self._data_i_appendnames    = np.load(data_i_appendnames_path, allow_pickle=True).item()
            return True
        else:
            self._update_cluster_all()
    
    def __parse_cache_kw(self, **kw):
        assert all([isinstance(v, list) for v in kw.values()])
        assert all([len(v) == len(self) for v in kw.values()])
        kw_keys = list(kw.keys())
        kw_values = list(kw.values())
        return kw_keys, kw_values

    def save_cache(self, **kw):
        self.__cache.clear()
        data_i_list = list(self.keys())
        kw_keys, kw_values = self.__parse_cache_kw(**kw)
        for data_i in tqdm(data_i_list, desc="save cache for {}".format(self.directory)):
            read_kw = {k:v[data_i] for k, v in zip(kw_keys, kw_values)}
            elem = self.read(data_i, **read_kw)
            self.__cache[data_i] = elem
        serialize_object(os.path.join(self.directory, self.__cache_name), self.__cache)

    def load_cache(self):
        cache_path = os.path.join(self.directory, self.__cache_name)
        if os.path.exists(cache_path):
            self.__cache = deserialize_object(cache_path)
            return True
        else:
            return False
        
    def unzip_cache(self, force = False, **kw):
        '''
        unzip the cache to a dict
        '''
        assert os.path.exists(os.path.join(self.directory, self.__data_i_dir_map_name))
        assert os.path.exists(os.path.join(self.directory, self.__data_i_appendnames_name))
        assert os.path.exists(os.path.join(self.directory, self.__cache_name))
        kw_keys, kw_values = self.__parse_cache_kw(**kw)
        if len(self.__cache) == 0:
            self.load_cache()
        progress = tqdm(zip(self._data_i_dir_map.keys(), self._data_i_appendnames.keys(), self.__cache.keys()), desc="unzip cache for {}".format(self.directory))
        for kd, ka, kc in progress:
            assert kd == ka == kc
            data_i = kd
            value = self.__cache[kd]
            appdir = self.data_i_dir_map[data_i]
            appname = self.data_i_appendnames[data_i][0]
            write_kw = {k:v[data_i] for k, v in zip(kw_keys, kw_values)}
            self.write(kd, value, appdir, appname, force=force, **write_kw)

class FileCluster(_DataCluster[FT, DCT]):
    '''
    a cluster of multiple files, they may have different suffixes and i/o operations
    but they must be read/write together
    '''
    class SingleFile():
        def __init__(self, sub_path:str, read_func:Callable, write_func:Callable) -> None:
            self.sub_path = sub_path
            self.read_func:Callable = read_func
            self.write_func:Callable = write_func
            self.cluster:FileCluster = None

        @property
        def path(self):
            return os.path.join(self.cluster.directory, self.sub_path)

        def set_cluster(self, cluster:"FileCluster"):
            self.cluster = cluster

        def read(self):
            return self.read_func(self.path)
        
        def write(self, data):
            self.write_func(self.path, data)

    def __init__(self, format_obj: FT, sub_dir, register, *singlefile:SingleFile) -> None:
        super().__init__(format_obj, sub_dir, register, *singlefile)

    @property
    def files(self):
        warnings.warn("__files is private, it's recommended to use method:update_file, remove_file", ClusterNotRecommendWarning)
        return self.__files
    
    @files.setter
    def files(self, value):
        warnings.warn("__files is private, it's recommended to use method:update_file, remove_file", ClusterNotRecommendWarning)
        self.__files = value

    @property
    def all_exist(self):
        for f in self.__files.values():
            if not os.path.exists(f.path):
                return False
        return True

    def __len__(self):
        super().__len__()
        return len(self.__files)

    def keys(self):
        super().keys()
        return list(self.__files.keys())
    
    def values(self) -> list[SingleFile]:
        super().values()
        return list(self.__files.values())
    
    def items(self):
        super().items()
        return self.keys(), self.values()

    def _read(self, data_i, *arg, **kwargs):
        super()._read(data_i, *arg, **kwargs)
        file_path = self.filter_data_i(data_i)
        return self.__files[file_path].read()
    
    def _write(self, data_i, value, *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)
        file_path = self.filter_data_i(data_i)
        return self.__files[file_path].write(value)

    def _clear(self, *arg, **kwargs):
        super()._clear(*arg, **kwargs)
        for fp in self.__files.keys():
            if os.path.exists(fp):
                os.remove(fp)

    def _copyto(self, dst: str, cover = False, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        os.makedirs(dst, exist_ok=True)
        for f in self.__files.values():
            dst_path = os.path.join(dst, f.sub_path)
            if os.path.exists(dst_path):
                if cover:
                    os.remove(dst_path)
                else:
                    raise FileExistsError(f"{dst_path} already exists, please set cover=True")
            shutil.copy(f.path, dst_path)

    def _init_attr(self, *singlefile:SingleFile, **kwargs):
        super()._init_attr(*singlefile, **kwargs)
        self.__files:dict[str, FileCluster.SingleFile] = {}

        for f in singlefile:
            f.set_cluster(self)
            self.update_file(f)

    def register_to_format(self):
        if self.register:
            self.format_obj.cluster_map[self.directory + str(id(self))] = self

    def filter_data_i(self, data_i):
        if isinstance(data_i, int):
            if data_i > len(self.__files):
                raise ClusterDataIOError("out of range")
            return list(self.__files.keys())[data_i]
        elif isinstance(data_i, str):
            if data_i not in self.__files:
                raise ClusterDataIOError(f"can't find {data_i} in {self.directory}")
            return data_i
        else:
            raise TypeError(f"unsupported type: {type(data_i)}")

    def copyto(self, dst: str, cover = False, *args, **kwargs):
        '''
        copy the whole directory to dst
        '''
        self._copyto(dst, cover, *args, **kwargs)

    def read_all(self):
        return [f.read() for f in self.values()]

    def write_all(self, values):
        assert len(values) == len(self), f"the length of value must be {len(self)}"
        for f, d in zip(self.values(), values):
            f.write(d)

    def remove_file(self, idx:Union[int, str]):
        if isinstance(idx, int):
            idx = list(self.keys())[idx]
        self.__files.pop(idx)

    def update_file(self, singlefile:SingleFile):
        singlefile.set_cluster(self)
        self.__files[singlefile.path] = singlefile

    def paths(self):
        return list(self.keys())

class IntArrayDictElement(Elements[FT, dict[int, np.ndarray]]):
    def __init__(self, format_obj: FT, sub_dir:str, array_shape:tuple[int], array_fmt:str = "", register=True, filllen=6, fillchar='0') -> None:
        super().__init__(format_obj, sub_dir, register, self._read_func, self._write_func, ".txt", filllen, fillchar)
        self.array_shape:tuple[int] = array_shape
        self.array_fmt = array_fmt if array_fmt else "%.4f"
    
    def _to_dict(self, array:np.ndarray)->dict[int, np.ndarray]:
        '''
        array: np.ndarray [N, 5]
        '''
        dict_ = {}
        for i in range(array.shape[0]):
            dict_[int(array[i, 0])] = array[i, 1:].reshape(self.array_shape)
        return dict_

    def _from_dict(self, dict_:dict[int, np.ndarray]):
        '''
        dict_: dict[int, np.ndarray]
        '''
        array = []
        for i, (k, v) in enumerate(dict_.items()):
            array.append(
                np.concatenate([np.array([k]).astype(v.dtype), v.reshape(-1)])
                )
        array = np.stack(array)
        return array

    def _read_format(self, array:np.ndarray, **kw):
        return array

    def _write_format(self, array:np.ndarray, **kw):
        return array

    def _read_func(self, path, **kw):
        raw_array = np.loadtxt(path, dtype=np.float32)
        if len(raw_array.shape) == 1:
            raw_array = np.expand_dims(raw_array, 0)
        raw_array = self._read_format(raw_array, **kw)
        intarraydict = self._to_dict(raw_array)
        return intarraydict

    def _write_func(self, path, intarraydict:dict[int, np.ndarray], **kw):
        raw_array = self._from_dict(intarraydict)
        raw_array = self._write_format(raw_array, **kw)
        np.savetxt(path, raw_array, fmt=self.array_fmt, delimiter='\t')
        
class DatasetFormatMode(enumerate):
    NORMAL = 0
    ONLY_CACHE = 1

class DatasetFormat(WriteController, ABC, Generic[DST]):
    """
    # Dataset Format
    -----
    A dataset manager, support mutiple data types.
    It is useful if you have a series of different data. 
    For example, a sample contains an image and a set of bounding boxes. 
    There is a one-to-one correspondence between them, and there are multiple sets of such data.

    properties
    ----
    * inited : bool, if the dataset has been inited
    * updated : bool, if the dataset has been updated, it can bes set.
    * directory : str, the root directory of the dataset
    * incomplete : bool, the last writing process of the dataset has not been completed
    * clusters : list[_DataCluster], all clusters of the dataset
    * opened_clusters : list[_DataCluster], all opened clusters of the dataset
    * jsondict_map : dict[str, JsonDict], the map of jsondict
    * elements_map : dict[str, Elements], the map of elements
    * files_map : dict[str, FileCluster], the map of fileclusters
    * data_num : int, the number of data in the dataset
    * data_i_upper : int, the max index of the iterator

    virtual function
    ----
    * read_one: Read one piece of data

    recommended to rewrite
    ----
    * _init_clusters: init the clusters
    * _write_jsondict: write one piece of data to jsondict
    * _write_elementss: write one piece of data to elements
    * _write_files: write one piece of data to files
    * _update_dataset: update the dataset, it should be called when the dataset is updated

    not necessary to rewrite
    ----
    * update_dataset: update the dataset, it should be called when the dataset is updated
    * read_from_disk: read all data from disk as a generator
    * write_to_disk: write one piece of data to disk
    * start_writing: start writing
    * stop_writing: stop writing    
    * clear: clear all data of the dataset
    * close_all: close all clusters
    * open_all : open all clusters
    * set_all_readonly: set all file streams to read only
    * set_all_writable: set all file streams to writable
    * get_element_paths_of_one: get the paths of one piece of data
    * __getitem__: get one piece of data
    * __setitem__: set one piece of data
    * __iter__: return the iterator of the dataset

    example
    -----
    * read

    df1 = DatasetFormat(directory1) 

    df2 = DatasetFormat(directory2) 

    for data in self.read_from_disk(): 
        ...

    * write_to_disk : 1

    df2.write_to_disk(data) × this is wrong

    with df2.writer:  # use context manager
    
        df2.write_to_disk(data)

    df2.clear()

    * write_to_disk : 2

    df2.start_writing()

    df2.write_to_disk(data)

    df2.stop_writing()
    
    * write_one

    df2.write_one(data_i, data)
    '''

    """
    DATAFRAME = "data_frame.csv"

    class __ClusterMap(dict):
        def __init__(self, format_obj:"DatasetFormat", *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.format_obj = format_obj

        def set_update(self):
            self.format_obj.updated = True
            if self.format_obj.inited:
                self.format_obj.update_dataset()

        def __setitem__(self, __key: Any, __value: Any) -> None:
            self.set_update()
            return super().__setitem__(__key, __value)
        
        def update(self, __m, **kwargs: Any) -> None:
            self.set_update()
            return super().update(__m, **kwargs)
    
        def setdefault(self, __key: Any, __default: Any = ...) -> Any:
            self.set_update()
            return super().setdefault(__key, __default)

    def __init__(self, directory, clear_incomplete = False, init_mode = DatasetFormatMode.NORMAL) -> None:
        super().__init__()
        ABC.__init__(self)
        self.__inited = False # if the dataset has been inited
        
        self.directory:str = directory
        self._updated = False
        self._data_num = 0
        self._data_i_upper = 0        

        print(f"initializing dataset: {self.__class__.__name__} at {self.directory} ...")
        os.makedirs(self.directory, exist_ok=True)

        self.incomplete = os.path.exists(self.get_mark_file())
        if self.incomplete:
            if clear_incomplete:
                pass
            else:
                tip = f"the last writing process of the dataset:{self.directory} has not been completed, \
                    if you want to clear all data, input 'y', else the program will exit: "
                print("="*int(len(tip) / 2), '\n', tip, '\n', "="*int(len(tip) / 2))
                y = input()
                if y != 'y':
                    raise ValueError("the dataset is incomplete")   

        self.cluster_map:dict[str, _DataCluster] = self.__ClusterMap(self)

        self._init_clusters()

        self.update_dataset()

        self.load_data_frame()        
        if self.incomplete:
            os.remove(self.get_mark_file())
            self.incomplete = False



        # # if not init_mode == DatasetFormatMode.ONLY_CACHE:
        #     self.incomplete = os.path.exists(self.get_mark_file())
        #     if self.incomplete:
        #         if clear_incomplete:
        #             pass
        #         else:
        #             tip = f"the last writing process of the dataset:{self.directory} has not been completed, \
        #                 if you want to clear all data, input 'y', else the program will exit: "
        #             print("="*int(len(tip) / 2), '\n', tip, '\n', "="*int(len(tip) / 2))
        #             y = input()
        #             if y != 'y':
        #                 raise ValueError("the dataset is incomplete")   

        #     self.cluster_map:dict[str, _DataCluster] = self.__ClusterMap(self)

        #     self._init_clusters()

        #     self.update_dataset()
        #     if self.incomplete:
        #         os.remove(self.get_mark_file())
        #         self.incomplete = False

        self.__inited = True # if the dataset has been inited


    @property
    def inited(self):
        return self.__inited

    @property
    def updated(self):
        return self._updated
    
    @updated.setter
    def updated(self, value:bool):
        self._updated = bool(value)

    @property
    def clusters(self):
        clusters = list(self.cluster_map.values())
        return clusters

    @property
    def data_clusters(self):
        clusters = [x for x in self.clusters if not isinstance(x, FileCluster)]
        return clusters

    @property
    def opened_clusters(self):
        clusters = [x for x in self.clusters if not x.is_close()]
        return clusters

    @property
    def opened_data_clusters(self):
        clusters = [x for x in self.data_clusters if not x.is_close()]
        return clusters

    @property
    def jsondict_map(self):
        '''
        select the key-value pair whose value is JsonDict
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, JsonDict)}

    @property
    def elements_map(self):
        '''
        select the key-value pair whose value is Elements
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, Elements)}
    
    @property
    def filecluster_map(self):
        '''
        select the key-value pair whose value is FileCluster
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, FileCluster)}

    @property
    def data_num(self):
        return self._data_num
    
    @property
    def data_i_upper(self):
        return self._data_i_upper
        # return max([x.cluster_data_i_upper for x in self.opened_clusters])
        
    @abstractmethod
    def read_one(self, data_i, *arg, **kwargs) -> DST:
        pass

    def _init_clusters(self):
        pass    

    def _write_jsondict(self, data_i, data):
        pass

    def _write_elements(self, data_i, data):
        pass

    def _write_files(self, data_i, data):
        pass

    def _update_dataset(self, data_i = None):
        nums = [len(x) for x in self.opened_data_clusters]
        num = np.unique(nums)
        if len(num) > 1:
            raise ValueError("Unknown error, the numbers of different datas are not equal")
        elif len(num) == 1:
            self._data_num = int(num)
        else:
            self._data_num = 0
        try:
            self._data_i_upper = max([x.cluster_data_i_upper for x in self.opened_data_clusters])
        except ValueError:
            self._data_i_upper = 0
        if self._data_i_upper != self.data_num:
            warnings.warn(f"the data_i_upper of dataset:{self.directory} is not equal to the data_num, \
                          it means the the data_i is not continuous, this may cause some errors", ClusterParaWarning)
        # if self.data_i_upper - 1 >= 0 and hasattr(self, "data_frame"):
        #     self.add_to_data_frame(self.data_i_upper - 1)

    def write_one(self, data_i, data:DST, *arg, **kwargs):
        assert self.is_writing or all([not x.write_streamly for x in self.jsondict_map.values()]), \
            "write_one cannot be used when any jsondict's stream_dumping_json is True. \
                considering use write_to_disk instead"
        self._write_jsondict(data_i, data)
        self._write_elements(data_i, data)
        self._write_files(data_i, data)

        self.update_dataset(data_i)

    def save_elements_cache(self):
        for elem in self.elements_map.values():
            elem.save_cache()

    def set_elements_cachemode(self, mode:bool):
        for elem in self.elements_map.values():
            elem.cache_mode = bool(mode)

    def get_mark_file(self):
        return os.path.join(self.directory, ".dfsw")
    
    def log_to_mark_file(self, logtxt = ""):
        with open(self.get_mark_file(), 'a') as f:
            f.write(logtxt + '\n')

    def update_dataset(self, data_i = None, f = False):
        if self.updated or f:
            self._update_dataset(data_i)
            self.updated = False

    def read_from_disk(self, with_data_i = False):
        '''
        brief
        ----
        *generator
        Since the amount of data may be large, return one by one
        '''
        if not with_data_i:
            for i in self.data_frame.index:
                yield self.read_one(i)
        else:
            for i in self.data_frame.index:
                yield i, self.read_one(i)

    def write_to_disk(self, data:DST, data_i = -1):
        '''
        brief
        -----
        write elements immediately, write basejsoninfo to cache, 
        they will be dumped when exiting the context of self.writer
        
        NOTE
        -----
        For DatasetFormat, the write mode has only 'append'. 
        If you need to modify, please call 'DatasetFormat.clear' to clear all data, and then write again.
        '''
        if not self.is_writing:
            print("please call 'self.start_writing' first, here is the usage example:")
            print(extract_doc(DatasetFormat.__doc__, "example"))
            raise ValueError("please call 'self.start_writing' first")
        if data_i == -1:
            # self._updata_data_num()        
            data_i = self.data_i_upper
        
        self.write_one(data_i, data)

    def start_writing(self, overwrite_allowed = False):
        super().start_writing(overwrite_allowed)
        print(f"start to write to {self.directory}")
        with open(self.get_mark_file(), 'w'): # create a file to mark that the DatasetFormat is writing
            pass
        self.close_all(False)
        self.set_all_readonly(False)
        self.set_all_overwrite_allowed(overwrite_allowed)

    def stop_writing(self):
        super().stop_writing()
        for jd in self.jsondict_map.values():
            jd.stop_writing()              
        os.remove(self.get_mark_file())
        self.set_all_readonly(True)
        self.set_all_overwrite_allowed(False)
        self.close_all()
        self.save_data_frame()

    def clear(self, ignore_warning = False, force = False):
        '''
        brief
        -----
        clear all data, defalut to ask before executing
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            if force:
                self.close_all(False)
                self.set_all_readonly(False)
                cluster_to_clear = self.clusters
            else:
                cluster_to_clear = self.opened_clusters

            for cluster in cluster_to_clear:
                cluster.clear(ignore_warning=True)

        self.set_all_readonly(True)

    def copyto(self, dst:str, cover = False):
        progress = tqdm(self.opened_clusters)
        for c in progress:
            progress.set_postfix({'copying': "{}:{}".format(c.__class__.__name__, c.directory)})            
            c.copyto(os.path.join(dst, c.sub_dir), cover=cover)

    def merge_from(self, src_dataset:DST):
        assert type(self) == type(src_dataset), "the type of src_dataset must be the same as self"
        raise NotImplementedError

    def zip_dict(self, ids:list[int], item:Union[Iterable, None, Any], func = lambda x: x):
        if item:
            processed = func(item)
            return as_dict(ids, processed)
        else:
            return None

    def close_all(self, value = True):
        for obj in list(self.elements_map.values()) + list(self.jsondict_map.values()):
            obj.close() if value else obj.open()

    def open_all(self, value = True):
        self.close_all(not value)

    def set_all_readonly(self, value = True):
        for obj in list(self.elements_map.values()) + list(self.jsondict_map.values()):
            obj.set_readonly(value)

    def set_all_writable(self, value = True):
        self.set_all_readonly(not value)

    def set_all_overwrite_allowed(self, value = True):
        for obj in list(self.elements_map.values()) + list(self.jsondict_map.values()):
            obj.set_overwrite_allowed(value)

    def get_element_paths_of_one(self, data_i:int):
        '''
        brief
        -----
        get all paths of a data
        '''
        paths = {}
        for elem in self.elements_map.values():
            paths[elem.sub_dir] = elem.auto_path(data_i)
        return paths
    
    def values(self):
        return self.read_from_disk()
    
    def keys(self):
        for i in self.data_frame.index:
            yield i

    def items(self):
        return self.read_from_disk(True)

    def __getitem__(self, data_i:Union[int, slice]):
        if isinstance(data_i, slice):
            # 处理切片操作
            start, stop, step = data_i.start, data_i.stop, data_i.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            def g():
                for i in range(start, stop, step):
                    yield self.read_one(i)
            return g()
        elif isinstance(data_i, int):
            # 处理单个索引操作
            return self.read_one(data_i)
        else:
            raise TypeError("Unsupported data_i type")

    def __setitem__(self, data_i:int, value):
        self.write_one(data_i, value)

    def __len__(self):
        return self.data_num

    def __iter__(self):
        return self.read_from_disk()
    
    def load_data_frame(self):
        data_frame_path = os.path.join(self.directory, self.__class__.__name__ + "-" + DatasetFormat.DATAFRAME)
        if os.path.exists(data_frame_path):
            self.data_frame = pd.read_csv(data_frame_path)
            # index_names = self.data_frame.iloc[:,0]
            # self.data_frame = self.data_frame.drop(self.data_frame.columns[0], axis = 1)
            # self.data_frame.index = index_names
        else:
            # self.init_data_frame()
            # self.data_frame.loc[-1, -1] = 'Left-Top'
            self.save_data_frame()
        return self.data_frame

    def init_data_frame(self):
        data = np.zeros((self.data_i_upper, len(self.data_clusters)), np.bool_)
        cols = [x.sub_dir for x in self.data_clusters]
        self.data_frame = pd.DataFrame(data, columns=cols)

        if self.data_num != self.data_i_upper:
            for data_i in tqdm(self.data_frame.index, desc="initializing data frame"):
                self.calc_data_frame(data_i)
        else:
            self.data_frame.loc[:] = True

    def save_data_frame(self):
        self.init_data_frame()
        if not self.data_frame.empty:
            data_frame_path = os.path.join(self.directory, self.__class__.__name__ + "-" + DatasetFormat.DATAFRAME)
            self.data_frame.to_csv(data_frame_path, index=False)

    def calc_data_frame(self, data_i):
        for cluster in self.data_clusters:
            self.data_frame.loc[data_i, cluster.sub_dir] = data_i in cluster.keys()

    def clear_invalid_data_i(self):
        raise NotImplementedError

    def add_to_data_frame(self, data_i:int):
        if data_i in self.data_frame.index:
            self.calc_data_frame(data_i)
        else:
            self.data_frame.loc[data_i] = [False for _ in range(len(self.data_frame.columns))]
            self.calc_data_frame(data_i)
 
#### Posture ####

class PostureDatasetFormat(DatasetFormat[ViewMeta]):

    def _init_clusters(self):
        self.labels_elements     = IntArrayDictElement(self, "labels", (4,), array_fmt="%8.8f")
        self.bbox_3ds_elements   = IntArrayDictElement(self, "bbox_3ds", (-1, 2), array_fmt="%8.8f") 
        self.landmarks_elements  = IntArrayDictElement(self, "landmarks", (-1, 2), array_fmt="%8.8f")
        self.extr_vecs_elements  = IntArrayDictElement(self, "trans_vecs", (2, 3), array_fmt="%8.8f")

    def read_one(self, data_i, appdir="", appname="", *arg, **kwargs) -> ViewMeta:
        super().read_one(data_i, appdir, appname, *arg, **kwargs)
        labels_dict:dict[int, np.ndarray] = self.labels_elements.read(data_i, appdir=appdir)
        extr_vecs_dict:dict[int, np.ndarray] = self.extr_vecs_elements.read(data_i, appdir=appdir)
        bbox_3d_dict:dict[int, np.ndarray] = self.bbox_3ds_elements.read(data_i, appdir=appdir)
        landmarks_dict:dict[int, np.ndarray] = self.landmarks_elements.read(data_i, appdir=appdir)
        return ViewMeta(color=None,
                        depth=None,
                        masks=None,
                        extr_vecs = extr_vecs_dict,
                        intr=None,
                        depth_scale=None,
                        bbox_3d = bbox_3d_dict, 
                        landmarks = landmarks_dict,
                        visib_fract=None,
                        labels=labels_dict)

    def _write_elements(self, data_i: int, viewmeta: ViewMeta, appdir="", appname=""):
        self.labels_elements.write(data_i, viewmeta.labels, appdir=appdir, appname=appname)
        self.bbox_3ds_elements.write(data_i, viewmeta.bbox_3d, appdir=appdir, appname=appname)
        self.landmarks_elements.write(data_i, viewmeta.landmarks, appdir=appdir, appname=appname)
        self.extr_vecs_elements.write(data_i, viewmeta.extr_vecs, appdir=appdir, appname=appname)

    def calc_by_base(self, mesh_dict:dict[int, MeshMeta], overwitre = False):
        '''
        brief
        -----
        calculate data by base data, see ViewMeta.calc_by_base
        '''
        with self.writer:
            self.set_overwrite_allowed(True)
            for i in range(self.data_num):
                viewmeta = self.read_one(i)
                viewmeta.calc_by_base(mesh_dict, overwitre=overwitre)
                self.write_to_disk(viewmeta, i)
            self.set_overwrite_allowed(False)

class CacheElements(Elements[PostureDatasetFormat, ViewMeta]):
    '''
    cache viewmeta as npy
    ----
    cache viewmeta as npy to speed up the loading process by about 4 times, \n
    At the same time, the space occupied by the file will increase by about 600%
    
    '''
    def __init__(self, format_obj, sub_dir, filllen=6, fillchar='0') -> None:
        super().__init__(format_obj, sub_dir, False, None, None, ".npy", filllen, fillchar)
        self.read_func = self._read_func
        self.write_func = self._write_func

        self.cvtMask_uint8 = False
    
    def _init_attr(self, *args, **kwargs):
        super()._init_attr(*args, **kwargs)
        self.read_func = self._read_func
        self.write_func = self._write_func

    def _get_from_meta(self, meta:dict, name):
        '''
        get value from meta dict
        '''
        value = meta[name]
        return value

    def _read_func(self, path):
        meta = np.load(path,  allow_pickle= True).item()
        ids         = self._get_from_meta(meta, 'ids')
        color         = self._get_from_meta(meta, 'color')
        depth       = self._get_from_meta(meta, 'depth')
        mask_dict   = self._decompress_mask(ids, self._get_from_meta(meta, 'cprsd_mask'), self.cvtMask_uint8)
        extr_vecs   = self._zip_dict(ids, self._get_from_meta(meta, 'extr_vecs'))
        intr        = self._get_from_meta(meta, 'intr')
        depth_scale = self._get_from_meta(meta, 'depth_scale')
        bbox_3d     = self._zip_dict(ids, self._get_from_meta(meta, 'bbox_3d'))
        landmarks   = self._zip_dict(ids, self._get_from_meta(meta, 'landmarks'))
        visib_fract = self._zip_dict(ids, self._get_from_meta(meta, 'visib_fract'))

        viewmeta = ViewMeta(color, depth, mask_dict, extr_vecs, intr, depth_scale, bbox_3d, landmarks, visib_fract)
        return viewmeta

    def _write_func(self, path, viewmeta:ViewMeta):
        color = viewmeta.color
        depth = viewmeta.depth
        masks = viewmeta.masks
        ids, cprsd_mask = self._compress_mask(masks)
        ids, extr_vecs = self._split_dict(viewmeta.extr_vecs)
        intr = viewmeta.intr
        depth_scale = viewmeta.depth_scale
        ids, bbox_3d = self._split_dict(viewmeta.bbox_3d)
        ids, landmarks = self._split_dict(viewmeta.landmarks)
        ids, visib_fract = self._split_dict(viewmeta.visib_fract)
        np.save(path, 
                 {"ids":ids, 
                 "color":color, 
                 "depth":depth, 
                 "cprsd_mask":cprsd_mask, 
                 "extr_vecs":extr_vecs, "intr":intr, 
                 "depth_scale":depth_scale, "bbox_3d":bbox_3d, 
                 "landmarks":landmarks, "visib_fract":visib_fract})
        
    @staticmethod
    def _zip_dict(ids:np.ndarray, array:np.ndarray) -> dict[int, np.ndarray]:
        '''
        generate a dict from ids and array
        '''
        if array is None:
            return None
        dict_ = dict(zip(ids, array))
        return dict_

    @staticmethod
    def _split_dict(dict_:dict[int, np.ndarray]):
        '''
        split a dict into ids and array
        '''
        if dict_ is None:
            return None, None
        return np.array(list(dict_.keys())), np.array(list(dict_.values()))

    @staticmethod
    def _compress_mask(mask_dict: dict[int, np.ndarray]):
        '''
        compress mask dict into ids and cprsd_mask \n
        a serial of mask will be compressed into a single array by bit operation
        '''
        if mask_dict is None:
            return None
        length = len(mask_dict)
        if length <= 8:
            dtype = np.uint8
        elif length <= 16:
            dtype = np.uint16
        elif length <= 32:
            dtype = np.uint32
        elif length <= 64:
            dtype = np.uint64
        elif length <= 128:
            dtype = np.uint128
        else:
            dtype = np.uint256
        mask_list = np.array(list(mask_dict.values()))
        mask_list = mask_list & 1
        maks_ids = np.array(list(mask_dict.keys()))
        cprsd_mask = np.zeros((mask_list[0].shape[0], mask_list[0].shape[1]), dtype=dtype)
        for shift, m in enumerate(mask_list):
            m = m << shift        
            cprsd_mask = np.bitwise_or(cprsd_mask, m)    
        return maks_ids, cprsd_mask

    @staticmethod
    def _decompress_mask(ids:np.ndarray, masks:np.ndarray, cvtMask_uint8):
        '''
        decompress ids and cprsd_mask into mask dict \n
        a single array will be decompressed into a serial of mask by bit operation
        '''
        if masks is None:
            return None
        mask_dict = {}
        for i, id in enumerate(ids):
            mask = (masks & (1 << i)).astype(np.bool8)
            if cvtMask_uint8:
                mask = mask.astype(np.uint8) * 255
            mask_dict[id] = mask
        return mask_dict

class LinemodFormat(PostureDatasetFormat):
    KW_CAM_K = "cam_K"
    KW_CAM_DS = "depth_scale"
    KW_CAM_VL = "view_level"
    KW_GT_R = "cam_R_m2c"
    KW_GT_t = "cam_t_m2c"
    KW_GT_ID = "obj_id"
    KW_GT_INFO_BBOX_OBJ = "bbox_obj"
    KW_GT_INFO_BBOX_VIS = "bbox_visib"
    KW_GT_INFO_PX_COUNT_ALL = "px_count_all"
    KW_GT_INFO_PX_COUNT_VLD = "px_count_valid"
    KW_GT_INFO_PX_COUNT_VIS = "px_count_visib" 
    KW_GT_INFO_VISIB_FRACT = "visib_fract"

    RGB_DIR = "rgb"
    DEPTH_DIR = "depth"
    MASK_DIR = "mask"
    GT_FILE = "scene_gt.json"
    GT_CAM_FILE = "scene_camera.json"
    GT_INFO_FILE = "scene_gt_info.json"
    
    class _MasksElements(Elements["LinemodFormat", dict[int, np.ndarray]]):
        def id_format(self, class_id):
            id_format = str(class_id).rjust(6, "0")
            return id_format

        def _read(self, data_i, *arg, **kwargs) -> dict[int, np.ndarray]:
            masks = {}
            appdir = ""
            for n, scene_gt in enumerate(self.format_obj.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask:np.ndarray = super()._read(data_i, appdir, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[id_] = mask
            return masks

        def _write(self, data_i, value: dict[int, ndarray], appdir="", appname="", *arg, **kwargs):
            for n, scene_gt in enumerate(self.format_obj.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask = value[id_]
                super().write(data_i, mask, appname=self.id_format(n))
            return super()._write(data_i, value, appdir, appname, *arg, **kwargs)

    def _init_clusters(self):
        super()._init_clusters()
        self.rgb_elements   = Elements(self,      self.RGB_DIR,
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.depth_elements = Elements(self,      self.DEPTH_DIR,    
                                       read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.masks_elements = self._MasksElements(self, self.MASK_DIR,     
                                                  read_func=lambda x:cv2.imread(x, cv2.IMREAD_GRAYSCALE),  
                                                  write_func=cv2.imwrite, 
                                                  suffix='.png')      

        self.scene_gt_dict              = JsonDict(self, self.GT_FILE)        
        self.scene_camera_dict          = JsonDict(self, self.GT_CAM_FILE)
        self.scene_gt_info_dict         = JsonDict(self, self.GT_INFO_FILE)

    def _write_elements(self, data_i:int, viewmeta:ViewMeta):
        super()._write_elements(data_i, viewmeta)
        self.rgb_elements.  write(data_i, viewmeta.color)
        self.depth_elements.write(data_i, viewmeta.depth)
        self.masks_elements.write(data_i, viewmeta.masks)

    def _write_jsondict(self, data_i:int, viewmeta:ViewMeta):
        super()._write_jsondict(data_i, viewmeta)
        gt_one_info = []
        for obj_id, trans_vecs in viewmeta.extr_vecs.items():
            posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])
            gt_one_info .append(
                {   LinemodFormat.KW_GT_R: posture.rmat.reshape(-1),
                    LinemodFormat.KW_GT_t: posture.tvec.reshape(-1),
                    LinemodFormat.KW_GT_ID: int(obj_id)})
        self.scene_gt_dict.write(self.data_num, gt_one_info)

        ###
        gt_cam_one_info = {self.KW_CAM_K: viewmeta.intr.reshape(-1), self.KW_CAM_DS: viewmeta.depth_scale, self.KW_CAM_VL: 1}
        self.scene_camera_dict.write(self.data_num, gt_cam_one_info)

        ### eg:
        # "0": 
        # [{"bbox_obj": [274, 188, 99, 106], 
        # "bbox_visib": [274, 188, 99, 106], 
        # "px_count_all": 7067, 
        # "px_count_valid": 7067, 
        # "px_count_visib": 7067, 
        # "visib_fract": 1.0}],
        gt_info_one_info = []
        bbox_2d = viewmeta.bbox_2d
        for obj_id in viewmeta.masks.keys():
            mask = viewmeta.masks[obj_id]
            bb = bbox_2d[obj_id]
            vf = viewmeta.visib_fract[obj_id]
            mask_count = int(np.sum(mask))
            mask_visib_count = int(mask_count * vf)
            gt_info_one_info.append({
                self.KW_GT_INFO_BBOX_OBJ: bb,
                self.KW_GT_INFO_BBOX_VIS: bb,
                self.KW_GT_INFO_PX_COUNT_ALL: mask_count, 
                self.KW_GT_INFO_PX_COUNT_VLD: mask_count, 
                self.KW_GT_INFO_PX_COUNT_VIS: mask_visib_count,
                self.KW_GT_INFO_VISIB_FRACT: vf
            })
        self.scene_gt_info_dict.write(self.data_num, gt_info_one_info)         

    def read_one(self, data_i, *arg, **kwargs):
        super().read_one(data_i, *arg, **kwargs)
        color     = self.rgb_elements.read(data_i)
        depth   = self.depth_elements.read(data_i)
        masks   = self.masks_elements.read(data_i)
        bbox_3d = self.bbox_3ds_elements.read(data_i)
        landmarks = self.landmarks_elements.read(data_i)
        intr           = self.scene_camera_dict[data_i][LinemodFormat.KW_CAM_K].reshape(3, 3)
        depth_scale    = self.scene_camera_dict[data_i][LinemodFormat.KW_CAM_DS]

        ids = [x[LinemodFormat.KW_GT_ID] for x in self.scene_gt_dict[data_i]]
        postures = [Posture(rmat =x[LinemodFormat.KW_GT_R], tvec=x[LinemodFormat.KW_GT_t]) for x in self.scene_gt_dict[data_i]]
        extr_vecs = [np.array([x.rvec, x.tvec]) for x in postures]
        extr_vecs_dict = as_dict(ids, extr_vecs)
        # visib_fract    = [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in self.scene_gt_info_dict[data_i]]
        visib_fract_dict = self.zip_dict(ids, self.scene_gt_info_dict[data_i], 
                                         lambda obj: [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in obj])
        return ViewMeta(color, depth, masks, 
                        extr_vecs_dict,
                        intr,
                        depth_scale,
                        bbox_3d,
                        landmarks,
                        visib_fract_dict)

class VocFormat(PostureDatasetFormat):
    DEFAULT_SPLIT_TYPE = ["default"]
    IMGAE_DIR = "JPEGImages"
    SPLIT_DIR = "ImageSets"

    KW_TRAIN = "train"
    KW_VAL = "val" 

    class cxcywhLabelElement(IntArrayDictElement):
        def __init__(self, format_obj: DatasetFormat, sub_dir: str, array_fmt: str = "", register=True, filllen=6, fillchar='0') -> None:
            super().__init__(format_obj, sub_dir, (4,), array_fmt, register, filllen=filllen, fillchar=fillchar)
            self.image_size_required = True
            self.__trigger = False

            self.default_image_size = None

        def ignore_warning_once(self):
            self.image_size_required = False
            self.__trigger = True

        def __reset_trigger(self):
            if self.__trigger:
                self.__trigger = False
                self.image_size_required = True

        def _read_format(self, labels: np.ndarray, image_size):
            if image_size is not None:
                bbox_2d = labels[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = VocFormat._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
                labels[:,1:] = bbox_2d   
            return labels         
        
        def _write_format(self, labels: np.ndarray, image_size):
            if image_size is not None:
                bbox_2d = labels[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = VocFormat._x1y1x2y2_2_normedcxcywh(bbox_2d, image_size)
                labels[:,1:] = bbox_2d
            return labels

        def read(self, data_i, appdir="", appname="", *arg, 
                 force = False, image_size = None, **kw)->dict[int, np.ndarray]:
            '''
            set_image_size() is supported be called before read() \n
            or the bbox_2d will not be converted from normed cxcywh to x1x2y1y2
            image_size: (w, h)
            '''
            if image_size is None:
                image_size = self.default_image_size
            if image_size is not None:
                return super().read(data_i, appdir, appname, *arg, force = force, image_size = image_size, **kw)
            else:
                if self.image_size_required:
                    warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2",
                                  ClusterParaWarning)
                self.__reset_trigger()
                return super().read(data_i, appdir, appname, *arg, image_size = image_size, **kw)
        
        def write(self, data_i, labels_dict:dict[int, np.ndarray], appdir="", appname="", *arg, 
                  force = False, image_size = None, **kw):
            '''
            set_image_size() is supported be called before write() \n
            or the bbox_2d will not be converted from x1x2y1y2 to normed cxcywh
            image_size: (w, h)
            '''
            if image_size is not None:
                return super().write(data_i, labels_dict, appdir, appname, *arg, force=force, image_size = image_size, **kw)
            else:
                if self.image_size_required:
                    warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh",
                                    ClusterParaWarning)
                self.__reset_trigger()                    
                return super().write(data_i, labels_dict, appdir, appname, *arg, force=force, image_size = image_size, **kw)    

    def __init__(self, directory, split_rate = 0.75, clear = False) -> None:
        super().__init__(directory, clear)

        self.split_rate = split_rate

        self.__split_mode:str = self.DEFAULT_SPLIT_TYPE[0]
        self.split_dir:str = os.path.join(self.directory, self.SPLIT_DIR)
        os.makedirs(self.split_dir, exist_ok=True)
        _split_mode_list:list[str] = remove_duplicates(os.listdir(self.split_dir) + self.DEFAULT_SPLIT_TYPE)
        self._spliters:dict[str, Spliter] = dict(zip(_split_mode_list, [Spliter(self.split_dir, m) for m in _split_mode_list]))

    def _init_clusters(self):
        super()._init_clusters()
        self.images_elements:Elements[VocFormat, np.ndarray]     = Elements(self, self.IMGAE_DIR,       
                                            read_func=cv2.imread, 
                                            write_func=cv2.imwrite,
                                            suffix = ".jpg")
        self.depth_elements      = Elements(self, "depths",       
                                            read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH), 
                                            write_func=cv2.imwrite, 
                                            suffix = '.png')
        self.masks_elements      = Elements(self, "masks",        
                                            read_func=lambda x: deserialize_image_container(deserialize_object(x), cv2.IMREAD_GRAYSCALE),
                                            write_func=lambda path, x: serialize_object(path, serialize_image_container(x)),  
                                            suffix = ".pkl")
        self.intr_elements       = Elements(self, "intr",
                                            read_func=loadtxt_func((3,3)), 
                                            write_func=savetxt_func("%8.8f"), 
                                            suffix = ".txt")
        self.depth_scale_elements         = Elements(self, "depth_scale",
                                            read_func=lambda path: float(loadtxt_func((1,))(path)), 
                                            write_func=savetxt_func("%8.8f"), 
                                            suffix = ".txt")
        self.visib_fract_elements= IntArrayDictElement(self, "visib_fracts", ())
        self.labels_elements     = self.cxcywhLabelElement(self, "labels",)

    def stop_writing(self):
        super().stop_writing()
        self.save_split()

    @property
    def split_mode(self):
        return self.__split_mode
    
    @split_mode.setter
    def split_mode(self, split_type:str):
        self.set_split_mode(split_type)

    @property
    def split_mode_list(self):
        return tuple(self._spliters.keys())

    def set_split_mode(self, split_type:str):
        assert split_type in self.split_mode_list, "split_type must be one of {}".format(self.split_mode_list)
        self.__split_mode = split_type

    def save_split(self):
        for s in self._spliters.values():
            s.save()

    @property
    def train_idx_array(self):
        spliter = self._spliters[self.__split_mode]
        assert spliter.split_for == spliter.SPLIT_FOR_TRAINING, "split_for must be {}".format(spliter.SPLIT_FOR_TRAINING)
        return self._spliters[self.__split_mode][self.KW_TRAIN]
    
    @property
    def val_idx_array(self):
        spliter = self._spliters[self.__split_mode]
        assert spliter.split_for == spliter.SPLIT_FOR_TRAINING, "split_for must be {}".format(spliter.SPLIT_FOR_TRAINING)
        return self._spliters[self.__split_mode][self.KW_VAL]

    @property
    def default_train_idx_array(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[0]][self.KW_TRAIN]
    
    @property
    def default_val_idx_array(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[0]][self.KW_VAL]

    def _get_split_of(self, split_type, sub_set:str = ""):
        if sub_set == "":
            dict_:dict[str, list] = self._spliters[split_type]
            # convert as dict[str, tuple]
            dict_ = {k:tuple(v) for k, v in dict_.items()}
            return dict_
        else:
            return tuple(self._spliters[split_type][sub_set])

    def get_data_type(self, data_i) -> dict[str, list[str]]:
        types = {}
        for s in self._spliters.values():
            if s.split_for == s.SPLIT_FOR_DATATYPE:
                types.update({s.split_mode_name: s.get_one_subtype(data_i)})
        return types

    def get_data_set(self, data_i) -> dict[str, list[str]]:
        sets = {}
        for s in self._spliters.values():
            if s.split_for == s.SPLIT_FOR_TRAINING:
                sets.update({s.split_mode_name: s.get_one_subtype(data_i)})
        return sets

    def get_default_set(self, data_i):
        if data_i in self.default_train_idx_array:
            sub_set = VocFormat.KW_TRAIN
        elif data_i in self.default_val_idx_array:
            sub_set = VocFormat.KW_VAL
        else:
            raise ValueError("can't find datas of index: {}".format(data_i))
        return sub_set
    
    def decide_default_set(self, data_i):
        try:
            return self.get_default_set(data_i)
        except ValueError:
            spliter = self._spliters[self.DEFAULT_SPLIT_TYPE[0]]
            return spliter.set_one_by_rate(data_i, self.split_rate)

    @staticmethod
    def _x1y1x2y2_2_normedcxcywh(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (x1, x2, y1, y2)]
        img_size: (w, h)
        '''

        # Calculate center coordinates (cx, cy) and width-height (w, h) of the bounding boxes
        x1, y1, x2, y2 = np.split(bbox_2d, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Normalize center coordinates and width-height by image size
        w_img, h_img = img_size
        cx_normed = cx / w_img
        cy_normed = cy / h_img
        w_normed = w / w_img
        h_normed = h / h_img

        # Return the normalized bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_normed = np.concatenate([cx_normed, cy_normed, w_normed, h_normed], axis=-1)
        return bbox_normed

        # lt = bbox_2d[..., :2]
        # rb = bbox_2d[..., 2:]

        # cx, cy = (lt + rb) / 2
        # w, h = rb - lt
        # # 归一化
        # cy, h = np.array([cy, h]) / img_size[0]
        # cx, w = np.array([cx, w]) / img_size[1]
        # return np.array([cx, cy, w, h])

    @staticmethod
    def _normedcxcywh_2_x1y1x2y2(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (cx, cy, w, h)]
        img_size: (w, h)
        '''

        # Unpack the normalized bounding box coordinates
        cx, cy, w, h = np.split(bbox_2d, 4, axis=-1)

        # Denormalize the center coordinates and width-height by image size
        w_img, h_img = img_size
        x1 = (cx - w / 2) * w_img
        y1 = (cy - h / 2) * h_img
        x2 = x1 + w * w_img
        y2 = y1 + h * h_img

        # Return the bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_2d = np.concatenate([x1, y1, x2, y2], axis=-1)
        return bbox_2d

    def _write_elements(self, data_i: int, viewmeta: ViewMeta):
        sub_set = self.decide_default_set(data_i)
        self.labels_elements.ignore_warning_once()
        super()._write_elements(data_i, viewmeta, appdir=sub_set)
        #
        self.images_elements.write(data_i, viewmeta.color, appdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, appdir=sub_set)
        #
        self.masks_elements.write(data_i, viewmeta.masks, appdir=sub_set)
        
        ###
        self.labels_elements.write(data_i, viewmeta.bbox_2d, appdir=sub_set, image_size = viewmeta.color.shape[:2][::-1]) # necessary to set image_size
        # labels = []
        # for id_, mask in viewmeta.masks.items():
        #     img_size = mask.shape
        #     point = np.array(np.where(mask))
        #     if point.size == 0:
        #         continue
        #     bbox_2d = viewmeta_bbox2d[id_]
        #     cx, cy, w, h = self._x1y1x2y2_2_normedcxcywh(bbox_2d, img_size)
        #     labels.append([id_, cx, cy, w, h])

        # self.labels_elements.write(data_i, labels, appdir=sub_set)

        self.intr_elements.write(data_i, viewmeta.intr, appdir=sub_set)
        self.depth_scale_elements.write(data_i, np.array([viewmeta.depth_scale]), appdir=sub_set)
        self.visib_fract_elements.write(data_i, viewmeta.visib_fract, appdir=sub_set)
    
    def read_one(self, data_i, *arg, **kwargs) -> ViewMeta:
        # 判断data_i属于train或者val
        sub_set = self.decide_default_set(data_i)

        self.labels_elements.ignore_warning_once()
        viewmeta = super().read_one(data_i, appdir=sub_set, *arg, **kwargs)
        # 读取
        color:np.ndarray = self.images_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.COLOR, color)
        #
        depth = self.depth_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.DEPTH, depth)
        #
        labels_dict = self.labels_elements.read(data_i, appdir=sub_set, image_size = color.shape[:2][::-1]) # {id: [cx, cy, w, h]}
        viewmeta.set_element(ViewMeta.LABELS, labels_dict)
        #
        masks_dict = self.masks_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.MASKS, masks_dict)
        #
        intr = self.intr_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.INTR, intr)
        #
        ds    = self.depth_scale_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.DEPTH_SCALE, ds)
        #
        visib_fract_dict = self.visib_fract_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.VISIB_FRACT, visib_fract_dict)

        return viewmeta

    def save_elements_cache(self):
        self.labels_elements.save_cache(image_size=[img.shape[:2][::-1] for img in self.images_elements])
        self.bbox_3ds_elements.save_cache()
        self.depth_scale_elements.save_cache()
        self.intr_elements.save_cache()
        self.landmarks_elements.save_cache()
        self.extr_vecs_elements.save_cache()
        self.visib_fract_elements.save_cache()

class _LinemodFormat_sub1(LinemodFormat):
    class _MasksElements(LinemodFormat._MasksElements):

        def _read(self, data_i, *arg, **kwargs) -> dict[int, ndarray]:
            masks = {}
            for n in range(100):
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[n] = mask
            return masks
        
        def _write(self, data_i, value: dict[int, ndarray], appdir="", appname="", *arg, **kwargs):
            for id_, mask in value.items():
                super().write(data_i, mask, appname=self.id_format(id_))
            return super()._write(data_i, value, appdir, appname, *arg, **kwargs)

    def __init__(self, directory, clear = False) -> None:
        super().__init__(directory, clear)
        self.rgb_elements   = Elements(self, "rgb", 
                                       read_func=cv2.imread,  
                                       write_func=cv2.imwrite, 
                                       suffix='.jpg')

    def read_one(self, data_i, *arg, **kwargs):
        viewmeta = super().read_one(data_i, *arg, **kwargs)

        for k in viewmeta.bbox_3d:
            viewmeta.bbox_3d[k] = viewmeta.bbox_3d[k][:, ::-1]
        for k in viewmeta.landmarks:
            viewmeta.landmarks[k] = viewmeta.landmarks[k][:, ::-1]
        viewmeta.depth_scale *= 1000

        return viewmeta

# class MixLinemod_VocFormat(VocFormat):
#     DEFAULT_SPLIT_TYPE = ["default", "posture", "reality", "basis"]
#     MODE_DETECTION = 0
#     MODE_POSTURE = 1

#     def __init__(self, directory, data_num=0, split_rate=0.75, clear=False) -> None:
#         super().__init__(directory, data_num, split_rate, clear)
#         self.mode = self.MODE_POSTURE

#     def _init_clusters(self):
#         super()._init_clusters()
#         # self.real_logfile       = "real_log.txt"
#         # self.sim_logfile        = "sim_log.txt"
#         # self.basic_logfile       = "basic_log.txt"    # basic data
#         # self.augment_logfile    = "augment_log.txt" # augment data
#         # self.posture_train_file = "posture_train.txt"
#         # self.posture_val_file   = "posture_val.txt"
#         # self.reality_split_files = FileCluster(self, "", True,
#         #                                         FileCluster.SingleFile(self.real_logfile,    self.loadsplittxt_func, self.savesplittxt_func),
#         #                                         FileCluster.SingleFile(self.sim_logfile, self.loadsplittxt_func, self.savesplittxt_func))
#         # self.posture_split_files = FileCluster(self, "", True,
#         #                                     FileCluster.SingleFile(self.posture_train_file, self.loadsplittxt_func, self.savesplittxt_func),
#         #                                     FileCluster.SingleFile(self.posture_val_file,   self.loadsplittxt_func, self.savesplittxt_func))
#         # self.basis_split_files = FileCluster(self, "", True,
#         #                                             FileCluster.SingleFile(self.basic_logfile,    self.loadsplittxt_func, self.savesplittxt_func),
#         #                                             FileCluster.SingleFile(self.augment_logfile, self.loadsplittxt_func, self.savesplittxt_func))
#         # self.load_real_sim_log()
#         # self.load_posture_log()
#         # self.load_basic_augment_log()

#     # def set_mode(self, mode:int):
#     #     assert mode in [self.MODE_DETECTION, self.MODE_POSTURE], "Unknown mode"
#     #     self.mode = mode

#     # @property
#     # def train_idx_array(self):
#     #     if self.mode == self.MODE_DETECTION:
#     #         return self.detection_train_idx_array
#     #     elif self.mode == self.MODE_POSTURE:
#     #         return self.posture_train_idx_array
#     #     else:
#     #         raise ValueError("Unknown mode")
        
#     # @property
#     # def val_idx_array(self):
#     #     if self.mode == self.MODE_DETECTION:
#     #         return self.detection_val_idx_array
#     #     elif self.mode == self.MODE_POSTURE:
#     #         return self.posture_val_idx_array
#     #     else:
#     #         raise ValueError("Unknown mode")

#     # def load_real_sim_log(self):
#     #     if self.reality_split_files.all_exist:
#     #         self.real_idx, self.sim_idx = self.reality_split_files.read_all()
#     #     else:
#     #         self.real_idx       = np.array([], dtype=np.int32)
#     #         self.sim_idx        = np.array([], dtype=np.int32)
    
#     # def load_posture_log(self):
#     #     if self.posture_split_files.all_exist:
#     #         self.posture_train_idx_array, self.posture_val_idx_array = self.posture_split_files.read_all()
#     #     else:
#     #         self.posture_train_idx_array = np.array([], dtype=np.int32)
#     #         self.posture_val_idx_array   = np.array([], dtype=np.int32)
    
#     # def load_basic_augment_log(self):
#     #     if self.basis_split_files.all_exist:
#     #         self.basic_idx, self.augment_idx = self.basis_split_files.read_all()
#     #     else:
#     #         self.basic_idx      = np.array([], dtype=np.int32)
#     #         self.augment_idx    = np.array([], dtype=np.int32)

#     def gen_posture_log(self, ratio = 0.85):
#         """
#         Only take ratio of the real data as the verification set
#         """
#         real_idx = self.real_idx.copy()
#         real_idx = np.random.shuffle(real_idx)
#         self.posture_val_idx_array = real_idx[:int(len(real_idx)*ratio)]
#         self.posture_train_idx_array = np.setdiff1d(
#             np.union1d(self.real_idx, self.sim_idx),
#             self.posture_val_idx_array
#             )

#     def record_data_type(self, data_i, is_real, is_basic):
#         def process_data(data_i, flag, array_1, array_2):
#             # 根据 is_real 选择要传入的数组
#             if flag:
#                 target_array = array_1
#                 opposite_array = array_2
#             else:
#                 target_array = array_2
#                 opposite_array = array_1

#             # 遍历 data_i 中的元素，检测是否包含在另一个数组中
#             if data_i not in target_array:
#                 target_array = np.append(target_array, data_i)
#             if data_i in opposite_array:
#                 opposite_array = np.setdiff1d(opposite_array, data_i)

#             if flag:
#                 array_1 = target_array
#                 array_2 = opposite_array
#             else:
#                 array_1 = opposite_array
#                 array_2 = target_array
#             return array_1, array_2
        
#         self.real_idx, self.sim_idx = process_data(data_i, is_real, self.real_idx, self.sim_idx)
#         self.basic_idx, self.augment_idx = process_data(data_i, is_basic, self.basic_idx, self.augment_idx)

#     def get_data_type(self, data_i):
#         is_real = data_i in self.real_idx
#         is_basic = data_i in self.basic_idx
#         return is_real, is_basic

#     def save_log(self):
#         self.reality_split_files.write_all((self.real_idx, self.sim_idx))
#         self.basis_split_files.write_all((self.basic_idx, self.augment_idx))
#         self.posture_split_files.write_all((self.posture_train_idx_array, self.posture_val_idx_array))

#     def clear(self, ignore_warning=False):
#         super().clear(ignore_warning)
#         self.real_idx = np.array([], dtype=np.int32)
#         self.sim_idx = np.array([], dtype=np.int32)
#         self.reality_split_files.clear()
#         # with open(self.real_logfile, "w") as f:
#         #     pass
#         # with open(self.sim_logfile, "w") as f:
#         #     pass

#     def stop_writing(self):
#         self.save_log()
#         super().stop_writing()
  
class Mix_VocFormat(VocFormat):
    DEFAULT_SPLIT_TYPE = ["default", "posture", "reality", "basis"]
    MODE_DETECTION = 0
    MODE_POSTURE = 1

    IMGAE_DIR = "JPEGImages"

    def __init__(self, directory, split_rate=0.75, clear=False) -> None:
        super().__init__(directory, split_rate, clear)

        self.split_mode = self.DEFAULT_SPLIT_TYPE[1]
        self._spliters[self.DEFAULT_SPLIT_TYPE[0]].split_for = Spliter.SPLIT_FOR_TRAINING
        self._spliters[self.DEFAULT_SPLIT_TYPE[1]].split_for = Spliter.SPLIT_FOR_TRAINING
        self._spliters[self.DEFAULT_SPLIT_TYPE[2]].split_for = Spliter.SPLIT_FOR_DATATYPE
        self._spliters[self.DEFAULT_SPLIT_TYPE[3]].split_for = Spliter.SPLIT_FOR_DATATYPE

        self._spliters[self.DEFAULT_SPLIT_TYPE[2]]._default_subtypes = ["real", "sim"]
        self._spliters[self.DEFAULT_SPLIT_TYPE[3]]._default_subtypes = ["basic", "augment"]

    @property
    def posture_train_idx_list(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[1]][self.KW_TRAIN]
    
    @property
    def posture_val_idx_list(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[1]][self.KW_VAL]

    @property
    def default_spliter(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[0]]
    
    @property
    def posture_spliter(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[1]]
    
    @property
    def reality_spliter(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[2]]
    
    @property
    def basis_spliter(self):
        return self._spliters[self.DEFAULT_SPLIT_TYPE[3]]

    def _init_clusters(self):
        super()._init_clusters()

    def gen_posture_log(self, ratio = 0.85):
        """
        Only take ratio of the real data as the verification set
        """
        assert self.reality_spliter.total_num == self.data_num, "reality_spliter.total_num != data_num"
        assert self.basis_spliter.total_num == self.data_num, "basis_spliter.total_num != data_num"
        
        real_idx = self.reality_spliter.get_subtype_idx_list(0).copy()
        np.random.shuffle(real_idx)
        self.posture_val_idx_list.clear()
        self.posture_val_idx_list.extend(real_idx[:int(len(real_idx)*ratio)])

        posture_train_idx_list = np.setdiff1d(
            np.union1d(self.reality_spliter.get_subtype_idx_list(0), self.reality_spliter.get_subtype_idx_list(1)),
            self.posture_val_idx_list
            ).astype(np.int32).tolist()
        self.posture_train_idx_list.clear()
        self.posture_train_idx_list.extend(posture_train_idx_list)

    def record_data_type(self, data_i, is_real, is_basic):
        self.reality_spliter.set_one(data_i, self.reality_spliter.subtypes[int(not is_real)])
        self.basis_spliter.set_one(data_i, self.basis_spliter.subtypes[int(not is_basic)])

    def get_data_type_as_bool(self, data_i):
        types = self.get_data_type(data_i)
        for k, v in types.items():
            types[k] = (v == self._spliters[k].subtypes[0])
        return types

class Spliter():
    SPLIT_FOR_TRAINING = 0
    SPLIT_FOR_DATATYPE = 1

    SPLIT_FOR_FILE = "__split_for.txt"

    def __init__(self, root:str, split_mode:str, split_for=None, default_subtypes = []) -> None:
        assert isinstance(default_subtypes, (list, tuple)), "default_subtypes must be a Iterable"

        self.root = root
        self.split_mode_name = split_mode
        os.makedirs(self.directory, exist_ok=True)
        if split_for is None:
            split_for_path = os.path.join(split_mode, self.SPLIT_FOR_FILE)
            if os.path.exists(split_for_path):
                split_for = self.loadsplittxt_func(os.path.join(split_mode, self.SPLIT_FOR_FILE))[0]
            else:
                split_for = self.SPLIT_FOR_TRAINING
        self.split_for = split_for

        if split_for == self.SPLIT_FOR_TRAINING:
            default_subtypes.extend([VocFormat.KW_TRAIN, VocFormat.KW_VAL])

        subs = os.listdir(self.directory)
        subtypes = [os.path.splitext(sub)[0] for sub in subs if sub != self.SPLIT_FOR_FILE]
        self.__default_subtypes = remove_duplicates(default_subtypes + subtypes)

        self.__exclusive = True
        self.load()
    
    @property
    def exclusive(self):
        return self.__exclusive
    
    @exclusive.setter
    def exclusive(self, value):
        if self.split_for == self.SPLIT_FOR_TRAINING:
            self.__exclusive = True
        else:
            self.__exclusive = bool(value)

    @property
    def directory(self):
        return os.path.join(self.root, self.split_mode_name)

    @property
    def subtypes(self):
        return tuple(self.split.keys())
    
    @property
    def _default_subtypes(self):
        return self.__default_subtypes
    
    @_default_subtypes.setter
    def _default_subtypes(self, value:Union[str, Iterable[str]]):
        if isinstance(value, str):
            value = [value]
        if isinstance(value, Iterable):
            self.__default_subtypes = tuple(remove_duplicates(value))
        for t in value:
            if t not in self.split:
                self.split[t] = []
        for t in list(self.split.keys()):
            if t not in value:
                self.split.pop(t)

    @property
    def total_num(self):
        return sum([len(v) for v in self.split.values()])

    def get_subtype_idx_list(self, subtype:Union[str, int]):
        if isinstance(subtype, str):
            return self.split[subtype]
        elif isinstance(subtype, int):
            return self.split[self.subtypes[subtype]]
        else:
            raise ValueError("subtype must be str or int")

    def get_file_path(self, subtype):
        return os.path.join(self.directory, subtype + ".txt")
    
    def loadsplittxt_func(self, path:str):
        if os.path.exists(path):
            return np.loadtxt(path).astype(np.int32).reshape(-1).tolist()
        else:
            return []
        
    def savesplittxt_func(self, path:str, value:Iterable):
        assert isinstance(value, Iterable), "split_dict value must be a Iterable"
        np.savetxt(path, np.array(value).astype(np.int32).reshape(-1, 1), fmt="%6d")

    def load(self):
        self.split:dict[str, list] = {}
        for subtype in self._default_subtypes:
            file_path = self.get_file_path(subtype)
            self.split[subtype] = self.loadsplittxt_func(file_path)
    
    def save(self):
        os.makedirs(self.directory, exist_ok=True)
        for subtype in self.subtypes:
            file_path = self.get_file_path(subtype)
            self.savesplittxt_func(file_path, self.split[subtype])
        self.savesplittxt_func(os.path.join(self.directory, self.SPLIT_FOR_FILE), [self.split_for])

    def _check_split_rate(self, split_rate:Union[float, Iterable[float], dict[str, float]]):
        assert len(self.subtypes) > 1, "len(subtypes) must > 1"
        if len(self.subtypes) == 2 and isinstance(split_rate, float):
            split_rate = (split_rate, 1 - split_rate)
        if isinstance(split_rate, dict):
            split_rate = tuple([split_rate[subtype] for subtype in self.subtypes])
        elif isinstance(split_rate, Iterable):
            split_rate = tuple(split_rate)
        else:
            raise ValueError("split_rate must be Iterable or dict[str, float], (or float if len(subtypes) == 2)")
        assert len(split_rate) == len(self.subtypes), "splite_rate must have {} elements".format(len(self.subtypes))
        return split_rate

    def gen(self, data_num:int, split_rate:Union[float, Iterable[float], dict[str, float]]):
        assert isinstance(data_num, int), "data_num must be int"
        split_rate = self._check_split_rate(split_rate)
        _s = np.arange(data_num, dtype=np.int32)
        np.random.shuffle(_s)
        for subtype, rate in zip(self.subtypes, split_rate):
            num = int(data_num * rate)
            self.split[subtype] = _s[:num]
            _s = _s[num:]
            
    def set_one(self, data_i, subtype, sort = False):
        if data_i not in self.split[subtype]:
            self.split[subtype].append(data_i)
        if self.exclusive:
            # remove from other subtypes
            for _subtype in self.subtypes:
                if _subtype != subtype and data_i in self.split[_subtype]:
                    self.split[_subtype].remove(data_i)
        if sort:
            self.split[subtype].sort()

    def set_one_by_rate(self, data_i, split_rate):
        split_rate = self._check_split_rate(split_rate)
        total_nums = [len(st) for st in self.split.values()]
        if sum(total_nums) == 0:
            # all empty, choose the first
            subtype_idx = 0
        else:
            rates = np.array(total_nums) / sum(total_nums)
            subtype_idx = 0
            for idx, r in enumerate(rates):
                if r <= split_rate[idx]:
                    subtype_idx = idx
                    break
        self.set_one(data_i, self.subtypes[subtype_idx])
        return self.subtypes[subtype_idx]

    def sort_all(self):
        for subtype in self.subtypes:
            self.split[subtype].sort()
    
    def get_one_subtype(self, data_i):
        subtypes = []
        for subtype in self.subtypes:
            if data_i in self.split[subtype]:
                subtypes.append(subtype)
        return subtypes

    def one_is_subtype(self, data_i, subtype):
        assert subtype in self.subtypes, "subtype must be one of {}".format(self.subtypes)
        return data_i in self.split[subtype]
    
    def __getitem__(self, key):
        return self.get_subtype_idx_list(key)

# class Spliter(FileCluster):
#     SPLIT_FOR_TRAINING = 0
#     SPLIT_FOR_DATATYPE = 1

#     SPLIT_FOR_FILE = "__split_for.txt"

#     def __init__(self, format_obj: Any, sub_dir, register, *singlefile: FileCluster.SingleFile, split_for=None, default_subtypes:list = []) -> None:
        
        
#         super().__init__(format_obj, sub_dir, register, *singlefile)

#         os.makedirs(os.path.join(self.directory, self.sub_dir))

#         if split_for is None:
#             split_for_path = os.path.join(self.split_mode, self.SPLIT_FOR_FILE)
#             if os.path.exists(split_for_path):
#                 split_for = self.loadsplittxt_func(os.path.join(self.split_mode, self.SPLIT_FOR_FILE))[0]
#             else:
#                 split_for = self.SPLIT_FOR_TRAINING
#         self.split_for = split_for

#         if split_for == self.SPLIT_FOR_TRAINING:
#             # default_subtypes.extend([VocFormat.KW_TRAIN, VocFormat.KW_VAL])
#             default_subtypes = [VocFormat.KW_TRAIN, VocFormat.KW_VAL]

#         subs = os.listdir(self.directory)
#         subtypes = [os.path.splitext(sub)[0] for sub in subs if sub != self.SPLIT_FOR_FILE]
#         self.__default_subtypes = remove_duplicates(default_subtypes + subtypes)

#         self.__exclusive = True
#         self.load()

#     @property 
#     def split_mode(self):
#         return self.sub_dir

#     @property
#     def exclusive(self):
#         return self.__exclusive
    
#     @exclusive.setter
#     def exclusive(self, value):
#         if self.split_for == self.SPLIT_FOR_TRAINING:
#             self.__exclusive = True
#         else:
#             self.__exclusive = bool(value)

#     @property
#     def subtypes(self):
#         return tuple(self.split.keys())
    
#     @property
#     def _default_subtypes(self):
#         return self.__default_subtypes
    
#     @_default_subtypes.setter
#     def _default_subtypes(self, value:Union[str, Iterable[str]]):
#         if isinstance(value, str):
#             value = [value]
#         if isinstance(value, Iterable):
#             self.__default_subtypes = tuple(remove_duplicates(value))
#         for t in value:
#             if t not in self.split:
#                 self.split[t] = []
#         for t in list(self.split.keys()):
#             if t not in value:
#                 self.split.pop(t)

#     @property
#     def total_num(self):
#         return sum([len(v) for v in self.split.values()])

#     def get_subtype_idx_list(self, subtype:Union[str, int]):
#         if isinstance(subtype, str):
#             return self.split[subtype]
#         elif isinstance(subtype, int):
#             return self.split[self.subtypes[subtype]]
#         else:
#             raise ValueError("subtype must be str or int")

#     def get_file_path(self, subtype):
#         return os.path.join(self.directory, subtype + ".txt")
    
#     def loadsplittxt_func(self, path:str):
#         if os.path.exists(path):
#             return np.loadtxt(path).astype(np.int32).reshape(-1).tolist()
#         else:
#             return []
        
#     def savesplittxt_func(self, path:str, value:Iterable):
#         assert isinstance(value, Iterable), "split_dict value must be a Iterable"
#         np.savetxt(path, np.array(value).astype(np.int32).reshape(-1, 1), fmt="%6d")

#     def load(self):
#         self.split:dict[str, list] = {}
#         for subtype in self._default_subtypes:
#             file_path = self.get_file_path(subtype)
#             self.split[subtype] = self.loadsplittxt_func(file_path)
    
#     def save(self):
#         os.makedirs(self.directory, exist_ok=True)
#         for subtype in self.subtypes:
#             file_path = self.get_file_path(subtype)
#             self.savesplittxt_func(file_path, self.split[subtype])
#         self.savesplittxt_func(os.path.join(self.directory, self.SPLIT_FOR_FILE), [self.split_for])

#     def _check_split_rate(self, split_rate:Union[float, Iterable[float], dict[str, float]]):
#         assert len(self.subtypes) > 1, "len(subtypes) must > 1"
#         if len(self.subtypes) == 2 and isinstance(split_rate, float):
#             split_rate = (split_rate, 1 - split_rate)
#         if isinstance(split_rate, dict):
#             split_rate = tuple([split_rate[subtype] for subtype in self.subtypes])
#         elif isinstance(split_rate, Iterable):
#             split_rate = tuple(split_rate)
#         else:
#             raise ValueError("split_rate must be Iterable or dict[str, float], (or float if len(subtypes) == 2)")
#         assert len(split_rate) == len(self.subtypes), "splite_rate must have {} elements".format(len(self.subtypes))
#         return split_rate

#     def gen(self, data_num:int, split_rate:Union[float, Iterable[float], dict[str, float]]):
#         assert isinstance(data_num, int), "data_num must be int"
#         split_rate = self._check_split_rate(split_rate)
#         _s = np.arange(data_num, dtype=np.int32)
#         np.random.shuffle(_s)
#         for subtype, rate in zip(self.subtypes, split_rate):
#             num = int(data_num * rate)
#             self.split[subtype] = _s[:num]
#             _s = _s[num:]
            
#     def set_one(self, data_i, subtype, sort = False):
#         if data_i not in self.split[subtype]:
#             self.split[subtype].append(data_i)
#         if self.exclusive:
#             # remove from other subtypes
#             for _subtype in self.subtypes:
#                 if _subtype != subtype and data_i in self.split[_subtype]:
#                     self.split[_subtype].remove(data_i)
#         if sort:
#             self.split[subtype].sort()

#     def set_one_by_rate(self, data_i, split_rate):
#         split_rate = self._check_split_rate(split_rate)
#         total_nums = [len(st) for st in self.split.values()]
#         if sum(total_nums) == 0:
#             # all empty, choose the first
#             subtype_idx = 0
#         else:
#             rates = np.array(total_nums) / sum(total_nums)
#             subtype_idx = 0
#             for idx, r in enumerate(rates):
#                 if r <= split_rate[idx]:
#                     subtype_idx = idx
#                     break
#         self.set_one(data_i, self.subtypes[subtype_idx])
#         return self.subtypes[subtype_idx]

#     def sort_all(self):
#         for subtype in self.subtypes:
#             self.split[subtype].sort()
    
#     def get_one_subtype(self, data_i):
#         subtypes = []
#         for subtype in self.subtypes:
#             if data_i in self.split[subtype]:
#                 subtypes.append(subtype)
#         return subtypes

#     def one_is_subtype(self, data_i, subtype):
#         assert subtype in self.subtypes, "subtype must be one of {}".format(self.subtypes)
#         return data_i in self.split[subtype]
    
#     def __getitem__(self, key):
#         return self.get_subtype_idx_list(key)




def serialize_object(file_path, obj:dict):
    # if os.path.splitext(file_path)[1] == '.pkl':
    #     file_path = os.path.splitext(file_path)[0] + ".npz"
    # np.savez(file_path, **obj)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

# 从文件反序列化对象
def deserialize_object(serialized_file_path):
    with open(serialized_file_path, 'rb') as file:
        elements = pickle.load(file)
        return elements

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result