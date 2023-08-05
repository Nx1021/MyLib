# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_keys
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
from open3d import geometry, utility, io
import os
import glob
import shutil
import pickle
import cv2
import time
import warnings

from abc import ABC, abstractmethod
from typing import Any, Union, Callable


from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .posture import Posture
from .mesh_manager import MeshMeta
from .utils import JsonIO, JSONDecodeError, extract_doc, _ignore_warning

def as_dict(ids, objs):
    if objs is None:
        return None
    else:
        return dict(zip(ids, objs))

def savetxt_func(fmt=...):
    return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

def loadtxt_func(shape:tuple[int]):
    return lambda path: np.loadtxt(path).reshape(shape)

class ClusterDataNotFoundError(Exception):
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

class _DataCluster(ABC):
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
    * _read_only: bool, Indicates whether the cluster is read-only or write-enabled.
    * changes_unsaved: bool, Indicates if any changes have been made to the cluster.
    * directory: str, Directory path for the cluster.

    # property
    -----
    * allow_overwitre: bool, Control the shielding of writing,
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
    - set_read_only: set the cluster as read-only or write-enabled.
    - is_read_only: check if the cluster is read-only.
    - _read_decorator: decorator function to handle reading operations when the cluster is closed.
    - _write_decorator: decorator function to handle writing operations when the cluster is closed or read-only.
    - clear: clear any data in the cluster. Subclasses may implement _clear.
    - read: read data from the cluster. Subclasses must implement _read.
    - write: write data to the cluster. Subclasses must implement _write.
    '''
    def __init__(self, format_obj: "DatasetFormat", sub_dir: str, register=True, *args, **kwargs) -> None:
        '''Initialize the data cluster with the provided format_obj, sub_dir, and registration flag.'''
        self.format_obj = format_obj
        self.sub_dir = sub_dir
        self.register = register
        self._incomplete = self.format_obj.incomplete
        self._error_to_load = False
        self._closed = True  # Indicates whether the cluster is closed or open.
        self._read_only = True  # Indicates whether the cluster is read-only or write-enabled.
        self._changed_since_opening = False  # Indicates if any changes have been made to the cluster.
        self.directory = os.path.join(format_obj.directory, self.sub_dir)  # Directory path for the cluster.
        self._data_i_upper = 0  

        self.__cluster_read_func:Callable =  self._read_decorator(self._read)
        self.__cluster_write_func:Callable = self._write_decorator(self._write)
        self.__cluster_clear_func:Callable = self._write_decorator(self._clear) 

        self._init_attr(*args, **kwargs)  # Initializes additional attributes specified by subclasses.
     
        if os.path.exists(self.directory) and not self._error_to_load:
            self.open()  # Opens the cluster for operation.
        else:
            self.close()

    @property
    def allow_overwitre(self):
        '''Property that returns whether the cluster format allows write operations.'''
        return self.format_obj.allow_overwitre

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
    def keys(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def _read(self, data_i, *arg, **kwargs):
        pass

    @abstractmethod
    def _write(self, data_i, value, *arg, **kwargs):
        pass

    @abstractmethod
    def _clear(self, *arg, **kwargs):
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
   
    def __getitem__(self, data_i):
        return self.read(data_i)
    
    def __setitem__(self, data_i, value):
        return self.write(data_i, value)

    def __iter__(self):
        return self.values()

    def open(self):
        '''Method to open the cluster for operation.'''
        self._closed = False  # Marks the cluster as open for operation.        
        if self._incomplete:
            read_only = self._read_only  # Stores the read-only flag.
            self.set_read_only(False)  # Sets the cluster as write-enabled.
            self.clear(ignore_warning=True)  # Clears any incomplete data if present.
            self._incomplete = False
            self.set_read_only(read_only)  # Restores the read-only flag.

    def close(self):
        '''Method to close the cluster, preventing further operations.'''
        self._changed_since_opening = False  # Resets the updated flag to false.
        self._closed = True  # Marks the cluster as closed.

    def is_close(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self._closed:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self._closed

    def set_read_only(self, read_only=True):
        '''Method to set the cluster as read-only or write-enabled.'''
        self._read_only = read_only

    def is_read_only(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self._read_only:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self._read_only

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
        def wrapper(self: "_DataCluster", data_i, *args, **kwargs):
            if self.is_close(with_warning=True):
                return None
            else:
                try:
                    rlt = func(data_i, *args, **kwargs)  # Calls the original function.
                except ClusterDataNotFoundError:
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
        def wrapper(self: "_DataCluster", data_i = None, value = None, *args, **kwargs):
            if self.is_close(with_warning=True) or self.is_read_only(with_warning=True):
                return None
            elif not self.allow_overwitre and data_i in self.keys():
                warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                              is not allowed to overwitre, any write operation will not be executed.",
                                ClusterIONotExecutedWarning)
                return None
            else:
                rlt = func(data_i, value, *args, **kwargs)  # Calls the original function.
                self.changed_since_opening = True  # Marks the cluster as updated after writing operations.
                if data_i is None:
                    self._update_cluster_all(*args, **kwargs)
                else:
                    self._update_cluster_inc(data_i, *args, **kwargs)
                return rlt
        return wrapper

    def clear(self, *, ignore_warning = False):
        '''
        Method to clear any data in the cluster. Subclasses may implement this method.
        * it is dargerous
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            self.__cluster_clear_func(self)
            return True
        else:
            return False

    def read(self, data_i, *args, **kwargs):
        '''
        Method to read data from the cluster. Subclasses must implement this method.
        '''
        return self.__cluster_read_func(self, data_i, *args, **kwargs)

    def write(self, data_i, value, *args, **kwargs):
        '''
        Method to write data to the cluster. Subclasses must implement this method.
        '''
        return self.__cluster_write_func(self, data_i, value, *args, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.format_obj}, {self.sub_dir}) at {hex(id(self))}"

class JsonDict(_DataCluster, dict):
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
    class _Placeholder():
        def __init__(self) -> None:
            pass

    class _StreamWriter():
        def __init__(self, jd:"JsonDict") -> None:
            self.jd = jd
            self.orig_jd_write_streamly = jd.write_streamly
            self.orig_jd_read_only      = jd.is_read_only()

        def __enter__(self):
            self.jd.write_streamly = True
            self.jd.open()
            self.jd.set_read_only(False)
            self.jd.stream.open()
            return self
    
        def __exit__(self, exc_type, exc_value, traceback):
            self.jd.write_streamly = self.orig_jd_write_streamly
            self.jd.set_read_only(self.orig_jd_read_only)
            self.jd.stop_stream_write()
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                return True

    def __init__(self, format_obj: "DatasetFormat", sub_dir: str, register = True) -> None:
        _DataCluster.__init__(self, format_obj, sub_dir, register)

    def __len__(self) -> int:
        super().__len__()
        return dict.__len__(self)

    def keys(self) -> dict_keys:
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
        return dict.__getitem__(self, data_i)

    def _write(self, data_i, value, *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)
        return dict.__setitem__(self, data_i, value)

    def _clear(self, *arg, **kwargs):
        with open(self.directory, 'w'):
            pass
        dict.clear(self)

    def _init_attr(self, *args, **kwargs):
        _DataCluster._init_attr(self, *args, **kwargs)
        if os.path.exists(self.directory):
            try:
                value = JsonIO.load_json(self.directory)
            except JSONDecodeError:
                self._error_to_load = True
                value = {}
        else:
            value = {}
        dict.__init__(self, value)
        self.stream = JsonIO.Stream(self.directory)
        self.write_streamly = False
        if self.register:
            if self.directory not in self.format_obj.jsondict_map:
                self.format_obj.jsondict_map.update({self.directory: self})
            else:
                self.format_obj.jsondict_map[self.directory].update(self) 

    def __iter__(self) -> Iterator:
        return _DataCluster.__iter__(self)

    # @_DataCluster.cluster_decorator(False)
    # def __getitem__(self, key):
    #     try:
    #         return super().__getitem__(key)
    #     except KeyError:
    #         return None
        
    # @_DataCluster.cluster_decorator(True)
    # def __setitem__(self, key, value):
    #     if (self.allow_overwitre or key not in self):
    #         super().__setitem__(key, value)

    @_DataCluster._write_decorator
    def update(self, *arg, **kw):
        warnings.warn(f"It's not recommended to use update() for {self.__class__.__name__} \
                      for it will call _update_cluster_default and spend more time. \
                      use write() or __setitem__ instead", ClusterNotRecommendWarning)
        super().update(*arg, **kw)

    def write(self, data_i, value, *args, **kwargs):
        if not self.write_streamly:
            rlt = super().write(data_i, value, *args, **kwargs)
            JsonIO.dump_json(self.directory, self)
            return rlt
        else:
            if self.stream.closed:
                self.stream.open()
            rlt = super().write(data_i, self._Placeholder(), *args, **kwargs)
            self.stream.write({data_i: value})
            return rlt
    
    def start_to_write_streamly(self):
        '''
        biref
        -----
        start to write streamly, it will return a StreamWriter object \n
        while writing streamly, the dict will be update by {data_i: _Placeholder()} \n
        it means that you can not read the new updated data, untill self.stream.close() is called \n

        example
        -----
        * 1 \n
        with jd.start_to_write_streamly() as sw:
            for i in range(10):
                sw.write(i)
        * 2 \n
        sw = jd.start_to_write_streamly()
        for i in range(10):
            sw.write(i)
        sw.stop_stream_write()
        '''
        return self._StreamWriter(self)

    def stop_stream_write(self):
        if self.write_streamly:
            self.stream.close()
            self._init_attr()


class Elements(_DataCluster):
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
                format_obj:"DatasetFormat",
                sub_dir,
                register = True,
                read_func:Callable = lambda x: None, 
                write_func:Callable = lambda x,y: None, 
                suffix:str = '.txt', 
                filllen = 6, 
                fillchar = '0') -> None:
        super().__init__(format_obj, sub_dir, register, read_func, write_func, suffix, filllen, fillchar)

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

    def _read(self, data_i, appdir = "", appname = "", *arg, **kwargs):
        super()._read(data_i, *arg, **kwargs)
        path = self.format_path(data_i, appdir=appdir, appname=appname)
        if not os.path.exists(path):
            path = self.auto_path(data_i)
        if os.path.exists(path):
            return self.read_func(path, *arg, **kwargs)
        else:
            raise ClusterDataNotFoundError(f"can't find {path}")
        
    def _write(self, data_i, value, appdir = "", appname = "", *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)

        path = self.format_path(data_i, appdir=appdir, appname=appname)
        dir_ = os.path.split(path)[0]
        os.makedirs(dir_, exist_ok=True)
        if value is not None:
            self.write_func(path, value, *arg, **kwargs)

    def _clear(self, *arg, **kwargs):
        super()._clear(*arg, **kwargs)
        shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def _init_attr(self, read_func, write_func, suffix, filllen, fillchar, *args, **kwargs):
        super()._init_attr(*args, **kwargs)
        self.filllen    = filllen
        self.fillchar   = fillchar
        self.suffix     = suffix
        if not self.suffix.startswith('.'):
            self.suffix = '.' + self.suffix
        self.read_func  = read_func
        self.write_func = write_func
        
        self._data_i_dir_map = {}
        self._data_i_appendnames = {}

        if self.register:
            self.format_obj.elements_map[self.directory] = self # register to format_obj

    def _update_cluster_inc(self, data_i, appdir = "", appname = "", *arg, **kwargs):
        self._data_i_dir_map[data_i] = appdir
        self._data_i_appendnames.setdefault(data_i, []).append(appname)

    def _update_cluster_all(self, *args, **kwargs):
        print(f'init {self.directory} data_i_dir_map, this may take a while...')
        self._data_i_dir_map.clear()
        self._data_i_appendnames.clear()
        for root, dirs, files in os.walk(self.directory):
            paths = glob.glob(root + "/*" + self.suffix)
            for path in paths:
                data_i, appdir, appname = self.parse_path(path)
                self._update_cluster_inc(data_i, appdir=appdir, appname=appname)
        # sort by data_i
        self._data_i_dir_map = dict(sorted(self._data_i_dir_map.items(), key=lambda x:x[0]))
        self._data_i_appendnames = dict(sorted(self._data_i_appendnames.items(), key=lambda x:x[0]))

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
    
    def open(self):
        if not os.path.exists(self.directory):
            print(f"Elements: {self.directory} is new, it will be created")
            os.makedirs(self.directory, exist_ok=True)
        return super().open()

    def read(self, data_i, appdir = "", appname = "", *arg, **kwarg):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        return super().read(data_i, appdir = appdir, appname = appname, *arg, **kwarg)

    def write(self, data_i, element, appdir = "", appname = "", *arg, **kwarg):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * element: the element to be written
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        return super().write(data_i, element, appdir = appdir, appname = appname, *arg, **kwarg)

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
                raise IndexError(f'idx {data_i} has more than one appendname: {appendnames}, its path is ambiguous. \
                                You must specify the appname by using method:read(data_i, appname=...)')
        if not return_app:
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            if os.path.exists(path):
                return path
            else:
                raise IndexError(f'idx {data_i} has no file in {self.directory}')
                return None
        else:
            return appdir, appname
        
class FileCluster(_DataCluster):
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

        def set_cluster(self, cluster:"FileCluster"):
            self.cluster = cluster
            self.path = os.path.join(cluster.directory, self.sub_path)

        def read(self):
            return self.read_func(self.path)
        
        def wrtie(self, data):
            self.write_func(self.path, data)

    def __init__(self, format_obj: "DatasetFormat", sub_dir = "", *singlefile:SingleFile) -> None:
        super().__init__(format_obj, sub_dir, False, *singlefile)

    @property
    def files(self):
        warnings.warn("__files is private, it's recommended to use method:update_file, remove_file", ClusterNotRecommendWarning)
        return self.__files
    
    @files.setter
    def files(self, value):
        warnings.warn("__files is private, it's recommended to use method:update_file, remove_file", ClusterNotRecommendWarning)
        self.__files = value

    @property
    def all_exits(self):
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
    
    def values(self):
        super().values()
        return self.read()
    
    def items(self):
        super().items()
        return self.keys(), self.values()

    def _read(self, data_i, *arg, **kwargs):
        super()._read(data_i, *arg, **kwargs)
        return [f.read() for f in self.__files.values()]
    
    def _write(self, data_i, value, *arg, **kwargs):
        assert len(value) == len(self.__files), f"the length of value must be {len(self.__files)}"
        super()._write(data_i, value, *arg, **kwargs)
        for f, d in zip(self.__files.values(), value):
            f.wrtie(d)

    def _clear(self, *arg, **kwargs):
        warnings.warn("it's not allowed to use clear() for FileCluster, \
                       please use write()", ClusterIONotExecutedWarning)
        return None

    def _init_attr(self, *singlefile:SingleFile, **kwargs):
        super()._init_attr(*singlefile, **kwargs)
        self.__files:dict[str, FileCluster.SingleFile] = {}

        for f in singlefile:
            f.set_cluster(self)
            self.update_file(f)

        if self.register:
            self.format_obj.files_map[self.directory] = self # register to format_obj

    def read(self, *args, **kwargs):
        return super().read(None, *args, **kwargs)
    
    def write(self, value, *args, **kwargs):
        return super().write(None, value, *args, **kwargs)

    def remove_file(self, idx:Union[int, str]):
        if isinstance(idx, int):
            idx = list(self.__files.keys())[idx]
        self.__files.pop(idx)

    def update_file(self, singlefile:SingleFile):
        singlefile.set_cluster(self)
        self.__files[singlefile.path] = singlefile

    def paths(self):
        return list(self.__files.keys())

class CacheElements(Elements):
    '''
    cache viewmeta as npy
    ----
    cache viewmeta as npy to speed up the loading process by about 4 times, \n
    At the same time, the space occupied by the file will increase by about 600%
    
    '''
    def __init__(self, format_obj: "DatasetFormat", sub_dir, filllen=6, fillchar='0') -> None:
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

class IntArrayDictElement(Elements):
    def __init__(self, format_obj: "DatasetFormat", sub_dir:str, array_shape:tuple[int], array_fmt:str = "", register=True, filllen=6, fillchar='0') -> None:
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

    def read(self, data_i, appdir="", appname="", **kw) -> dict[int, np.ndarray]:
        return super().read(data_i, appdir, appname, **kw)
    
    def write(self, data_i, raw_array_dict:dict[int, np.ndarray], appdir="", appname="", **kw):
        return super().write(data_i, raw_array_dict, appdir, appname, **kw)
        
class DatasetFormatMode(enumerate):
    NORMAL = 0
    ONLY_CACHE = 1

class DatasetFormat(ABC):
    '''
    # Dataset Format
    -----
    A dataset manager for 6D pose estimation, based on the general .viewmeta.ViewMeta for data reading and writing

    built-in data types
    -----
    * _Writer: write context manager, must call start_to_write method before writing
    * ClusterMap: a dict that can set_update when it is changed

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
    * read_from_disk: read all data from disk
    * write_to_disk: write one piece of data to disk
    * start_to_write: return a _Writer object
    * clear: clear all data of the dataset
    * close_all: close all file streams
    * set_all_read_only: set all file streams to read only
    * get_element_paths_of_one: get the paths of one piece of data

    example
    -----
    * read
    df1 = DatasetFormat(directory1) 
    df2 = DatasetFormat(directory2) 
    for data in self.read_from_disk(): 
        ...
    * write_to_disk 
    #df2.write_to_disk(data) × this is wrong 
    with df2.start_to_write():
        df2.write_to_disk(data)
    df2.clear()
    * write_one
    df2.write_one(data_i, data)

    '''
    class _Writer:
        '''
        write context manager
        -----
        call self.format_obj._dump_cache() when exit
        '''
        def __init__(self, format_obj:"DatasetFormat", overwitre = False) -> None:
            self.format_obj: DatasetFormat = format_obj
            self.format_obj.allow_overwitre = overwitre

        def __enter__(self):
            print(f"start to write to {self.format_obj.directory}")
            with open(self.mark_file(self.format_obj.directory), 'w'): # create a file to mark that the DatasetFormat is writing
                pass
            self.format_obj.close_all(False)
            self.format_obj.set_all_read_only(False)

            # for jd in self.format_obj.jsondict_map.values():
            #     if jd.write_streamly:
            #         jd.stream.open()
            return self
        
        def __exit__(self, exc_type, exc_value:Exception, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
                return False
            else:
                for jd in self.format_obj.jsondict_map.values():
                    jd.stream.close()                
                os.remove(self.mark_file(self.format_obj.directory))
                self.format_obj.set_all_read_only(True)
                return True
            
        @staticmethod
        def mark_file(directory):
            return os.path.join(directory, ".dfsw")
    
    class ClusterMap(dict):
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
        self.__inited = False # if the dataset has been inited
        
        self.directory:str = directory
        os.makedirs(self.directory, exist_ok=True)
        if not init_mode == DatasetFormatMode.ONLY_CACHE:
            self.incomplete = os.path.exists(self._Writer.mark_file(self.directory))
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
            self.writer:DatasetFormat._Writer = None       
            self.allow_overwitre = False  

            self.jsondict_map:dict[str, JsonDict] = self.ClusterMap(self)
            self.elements_map:dict[str, Elements] = self.ClusterMap(self)
            self.files_map:dict[str, FileCluster] = self.ClusterMap(self)

            self._update = False
            self._data_num = 0
            self._data_i_upper = 0

            self._init_clusters()

            self.update_dataset()
            if self.incomplete:
                os.remove(self._Writer.mark_file(self.directory))
                self.incomplete = False
        self.cache_elements = CacheElements(self, "cache")

        self.__inited = True # if the dataset has been inited

    @property
    def inited(self):
        return self.__inited

    @property
    def updated(self):
        return self._update
    
    @updated.setter
    def updated(self, value:bool):
        self._update = bool(value)

    @property
    def clusters(self):
        clusters = list(self.jsondict_map.values()) + list(self.elements_map.values()) + list(self.files_map.values())
        return clusters

    @property
    def opened_clusters(self):
        clusters = [x for x in self.clusters if not x.is_close()]
        return clusters

    @property
    def data_num(self):
        return self._data_num
    
    @property
    def data_i_upper(self):
        return self._data_i_upper
        return max([x.cluster_data_i_upper for x in self.opened_clusters])
        
    @abstractmethod
    def read_one(self, data_i, *arg, **kwargs):
        pass

    def write_one(self, data_i, data, *arg, **kwargs):
        assert self.writer is not None or all([not x.write_streamly for x in self.jsondict_map.values()]), \
            "write_one cannot be used when any jsondict's stream_dumping_json is True. \
                considering use write_to_disk instead"
        self._write_jsondict(data_i, data)
        self._write_elements(data_i, data)
        self._write_files(data_i, data)

        self.update_dataset(data_i)

    def _init_clusters(self):
        pass    

    def _write_jsondict(self, data_i, data):
        pass

    def _write_elements(self, data_i, data):
        pass

    def _write_files(self, data_i, data):
        pass

    def _update_dataset(self, data_i = None):
        nums = [len(x) for x in self.opened_clusters]
        num = np.unique(nums)
        if len(num) > 1:
            raise ValueError("Unknown error, the numbers of different datas are not equal")
        elif len(num) == 1:
            self._data_num = int(num)
        else:
            self._data_num = 0
        try:
            self._data_i_upper = max([x.cluster_data_i_upper for x in self.opened_clusters])
        except ValueError:
            self._data_i_upper = 0
        if self._data_i_upper != self.data_num:
            warnings.warn(f"the data_i_upper of dataset:{self.directory} is not equal to the data_num, \
                          it means the the data_i is not continuous, this may cause some errors", ClusterParaWarning)

    def update_dataset(self, data_i = None, f = False):
        if self.updated or f:
            self._update_dataset(data_i)
            self.updated = False

    def read_from_disk(self):
        '''
        brief
        ----
        *generator
        Since the amount of data may be large, return one by one
        '''
        for i in range(self.data_num):
            yield self.read_one(i)

    def write_to_disk(self, data, data_i = -1):
        '''
        brief
        -----
        write elements immediately, write basejsoninfo to cache, they will be dumped when exiting the context of self.writer
        
        NOTE
        -----
        For DatasetFormat, the write mode has only 'append'. 
        If you need to modify, please call 'DatasetFormat.clear' to clear all data, and then write again.
        '''
        if self.writer is None:
            print("please call 'self.start_to_write' first, here is the usage example:")
            print(extract_doc(DatasetFormat.__doc__, "example"))
            raise ValueError("please call 'self.start_to_write' first")
        if data_i == -1:
            # self._updata_data_num()        
            data_i = self.data_i_upper
        
        self.write_one(data_i, data)

    def start_to_write(self, overwitre = False):
        '''
        must be called before 'write_to_disk'
        '''
        if self.writer is None:
            self.writer = self._Writer(self, overwitre)
            return self.writer

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
                self.set_all_read_only(False)
                cluster_to_clear = self.clusters
            else:
                cluster_to_clear = self.opened_clusters

            for cluster in cluster_to_clear:
                cluster.clear(ignore_warning=True)
            #     if os.path.exists(path):
            #         if os.path.isdir(path):
            #             shutil.rmtree(path)
            #             os.makedirs(path)
            #         else:
            #             os.remove(path)
            # [x.clear(ignore_warning=True) for x in self.jsondict_map.values()]
        self.set_all_read_only(True)

    def close_all(self, value = True):
        for obj in list(self.elements_map.values()) + list(self.jsondict_map.values()):
            obj.close() if value else obj.open()

    def set_all_read_only(self, value = True):
        for obj in list(self.elements_map.values()) + list(self.jsondict_map.values()):
            obj.set_read_only(value)

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
    
    # def _parse_viewmeta_for_jsons(self, viewmeta:ViewMeta, data_i):
    #     parse = {}
    #     return parse

    # def _cache_source_info(self, parsed:dict):
    #     for path in self.jsondict_map.keys():
    #         if self.stream_dumping_json and not self.jsondict_map[path].is_close() and not self.jsondict_map[path].is_read_only():
    #             if path not in self._streams:
    #                 self._streams[path] = JsonIO.create_stream(path)                
    #             self._streams[path].write(parsed[path])

    #             keys = parsed[path].keys()
    #             self.jsondict_map[path].update(dict(zip(keys, [None for _ in keys])))
    #         else:
    #             self.jsondict_map[path].update(parsed[path])

    # def _updata_data_num(self):
    #     '''
    #     The total number of data of all types must be equal, otherwise an exception is thrown
    #     '''
    #     datas = list(self.jsondict_map.values()) + list(self.elements_map.values())
    #     datas = [x for x in datas if not x.is_close()]
    #     nums = [len(x) for x in datas]
    #     num = np.unique(nums)
    #     if len(num) > 1:
    #         raise ValueError("Unknown error, the numbers of different datas are not equal")
    #     elif len(num) == 1:
    #         self._data_num = int(num)
    #     else:
    #         self._data_num = 0

class PostureDatasetFormat(DatasetFormat):

    def _init_clusters(self):
        self.labels_elements     = IntArrayDictElement(self, "labels", (4,), array_fmt="%8.8f")
        self.bbox_3ds_elements   = IntArrayDictElement(self, "bbox_3ds", (-1, 2), array_fmt="%8.8f") 
        self.landmarks_elements  = IntArrayDictElement(self, "landmarks", (-1, 2), array_fmt="%8.8f")
        self.extr_vecs_elements  = IntArrayDictElement(self, "trans_vecs", (2, 3), array_fmt="%8.8f")

    def read_one(self, data_i, appdir="", appname="") -> ViewMeta:
        super().read_one(data_i, appdir, appname)
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
        with self.start_to_write():
            self.allow_overwitre = True
            for i in range(self.data_num):
                viewmeta = self.read_one(i)
                viewmeta.calc_by_base(mesh_dict, overwitre=overwitre)
                self.write_to_disk(viewmeta, i)
            self.allow_overwitre = False

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
    
    class _MasksElements(Elements):
        def id_format(self, class_id):
            id_format = "_" + str(class_id).rjust(6, "0")
            return id_format

        def read(self, data_i):
            masks = {}
            obj:LinemodFormat = self.format_obj
            for n, scene_gt in enumerate(obj.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[id_] = mask
            return masks
        
        def write(self, data_i, masks:dict[int, np.ndarray]):
            for n, mask in enumerate(masks.values()):
                super().write(data_i, mask, appname=self.id_format(n))

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

    def read_one(self, data_i):
        super().read_one(data_i)
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
        visib_fract    = [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in self.scene_gt_info_dict[data_i]]
        visib_fract_dict = as_dict(ids, visib_fract)
        return ViewMeta(color, depth, masks, 
                        extr_vecs_dict,
                        intr,
                        depth_scale,
                        bbox_3d,
                        landmarks,
                        visib_fract_dict)

class VocFormat(PostureDatasetFormat):
    KW_TRAIN = "train"
    KW_VAL = "val"

    class _Writer(DatasetFormat._Writer):
        def __init__(self, format_obj: "VocFormat", overwitre=False) -> None:
            super().__init__(format_obj, overwitre)
            self.format_obj: VocFormat = format_obj

        def __exit__(self, exc_type, exc_value: Exception, traceback):
            exit_return = super().__exit__(exc_type, exc_value, traceback)
            np.savetxt(self.format_obj.train_txt,   self.format_obj.detection_train_idx_array,    fmt = "%6d")
            np.savetxt(self.format_obj.val_txt,     self.format_obj.detection_val_idx_array,      fmt = "%6d")    
            return exit_return    

    class cxcywhLabelElement(IntArrayDictElement):
        def __init__(self, format_obj: DatasetFormat, sub_dir: str, array_fmt: str = "", register=True, filllen=6, fillchar='0') -> None:
            super().__init__(format_obj, sub_dir, (4,), array_fmt, register, filllen=filllen, fillchar=fillchar)

        def _read_format(self, labels: np.ndarray, image_size):
            bbox_2d = labels[:,1:] #[cx, cy, w, h]
            bbox_2d = VocFormat._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
            labels[:,1:] = bbox_2d   
            return labels         
        
        def _write_format(self, labels: np.ndarray, image_size):
            bbox_2d = labels[:,1:] #[cx, cy, w, h]
            bbox_2d = VocFormat._x1y1x2y2_2_normedcxcywh(bbox_2d, image_size)
            labels[:,1:] = bbox_2d
            return labels

        def read(self, data_i, appdir="", appname="", image_size = None, *, ignore_warning = False)->dict[int, np.ndarray]:
            '''
            set_image_size() is supported be called before read() \n
            or the bbox_2d will not be converted from normed cxcywh to x1x2y1y2
            '''
            if image_size is not None:
                return super().read(data_i, appdir, appname, image_size = image_size)
            else:
                if not ignore_warning:
                    warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2",
                                  ClusterParaWarning)
                return None
        
        def write(self, data_i, labels_dict:dict[int, np.ndarray], appdir="", appname="", image_size = None):
            '''
            set_image_size() is supported be called before write() \n
            or the bbox_2d will not be converted from x1x2y1y2 to normed cxcywh
            '''
            if image_size is not None:
                return super().write(data_i, labels_dict, appdir, appname, image_size = image_size)
            else:
                warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh",
                                  ClusterParaWarning)

    def __init__(self, directory, data_num = 0, split_rate = 0.75, clear = False) -> None:
        super().__init__(directory, clear)

        self.train_txt = os.path.join(self.directory,   VocFormat.KW_TRAIN + ".txt")
        self.val_txt = os.path.join(self.directory,     VocFormat.KW_VAL + ".txt")
        self.split_rate = split_rate
        self.get_split_file(data_num, split_rate)

    def _init_clusters(self):
        super()._init_clusters()
        self.images_elements     = Elements(self, "images",       
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
                                            
    def get_split_file(self, data_num, split_rate):
        create = False
        if data_num == 0:
            # 存在则读取，不存在则创建
            if os.path.exists(self.train_txt):
                self.detection_train_idx_array = np.loadtxt(self.train_txt).astype(np.int32)
                self.detection_val_idx_array   = np.loadtxt(self.val_txt).astype(np.int32)
            else:
                create = True
        else:
            create = True

        if create:
            data_i_list = list(range(data_num))
            np.random.shuffle(data_i_list) 
            self.detection_train_idx_array = np.array(data_i_list[: int(data_num*split_rate)]).astype(np.int32).reshape(-1)
            self.detection_val_idx_array   = np.array(data_i_list[int(data_num*split_rate): ]).astype(np.int32).reshape(-1)
            np.savetxt(self.train_txt, self.detection_train_idx_array, fmt = "%6d")
            np.savetxt(self.val_txt, self.detection_val_idx_array, fmt = "%6d")

    @property
    def train_idx_array(self):
        return self.detection_train_idx_array
    
    @property
    def val_idx_array(self):
        return self.detection_val_idx_array

    def decide_set(self, data_i, create_if_not_exist = False):
        if data_i in self.detection_train_idx_array:
            sub_set = VocFormat.KW_TRAIN
        elif data_i in self.detection_val_idx_array:
            sub_set = VocFormat.KW_VAL
        else:
            if create_if_not_exist:
                train_num = len(self.detection_train_idx_array)
                val_num = len(self.detection_val_idx_array)
                if train_num/self.split_rate > val_num/(1 - self.split_rate):
                    sub_set = VocFormat.KW_VAL
                    self.detection_val_idx_array = np.append(self.detection_val_idx_array, data_i)
                else:
                    sub_set = VocFormat.KW_TRAIN
                    self.detection_train_idx_array = np.append(self.detection_train_idx_array, data_i)
            else:
                raise ValueError("can't find datas of index: {}".format(data_i))
        return sub_set

    @staticmethod
    def _x1y1x2y2_2_normedcxcywh(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (x1, x2, y1, y2)]
        img_size: (h, w)
        '''

        # Calculate center coordinates (cx, cy) and width-height (w, h) of the bounding boxes
        x1, y1, x2, y2 = np.split(bbox_2d, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Normalize center coordinates and width-height by image size
        h_img, w_img = img_size
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
        img_size: (h, w)
        '''

        # Unpack the normalized bounding box coordinates
        cx, cy, w, h = np.split(bbox_2d, 4, axis=-1)

        # Denormalize the center coordinates and width-height by image size
        h_img, w_img = img_size
        x1 = (cx - w / 2) * w_img
        y1 = (cy - h / 2) * h_img
        x2 = x1 + w * w_img
        y2 = y1 + h * h_img

        # Return the bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_2d = np.concatenate([x1, y1, x2, y2], axis=-1)
        return bbox_2d

    def _write_elements(self, data_i: int, viewmeta: ViewMeta):
        sub_set = self.decide_set(data_i, create_if_not_exist = True)
        _ignore_warning(super()._write_elements, ClusterParaWarning)(data_i, viewmeta, appdir=sub_set)
        #
        self.images_elements.write(data_i, viewmeta.color, appdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, appdir=sub_set)
        #
        self.masks_elements.write(data_i, viewmeta.masks, appdir=sub_set)
        
        ###
        self.labels_elements.write(data_i, viewmeta.bbox_2d, appdir=sub_set, image_size = viewmeta.color.shape[:2]) # necessary to set image_size
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
    
    def read_one(self, data_i) -> ViewMeta:
        # 判断data_i属于train或者val
        sub_set = self.decide_set(data_i)

        viewmeta = _ignore_warning(super().read_one)(data_i, appdir=sub_set)
        # 读取
        color:np.ndarray = self.images_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.COLOR, color)
        #
        depth = self.depth_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.DEPTH, depth)
        #
        labels_dict = self.labels_elements.read(data_i, appdir=sub_set, image_size = color.shape[:2]) # {id: [cx, cy, w, h]}
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

class _LinemodFormat_sub1(LinemodFormat):
    class _MasksElements(Elements):
        def id_format(self, class_id):
            id_format = "_" + str(class_id).rjust(6, "0")
            return id_format

        def read(self, data_i):
            masks = {}
            for n in range(100):
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[n] = mask
            return masks
        
        def write(self, data_i, masks:dict[int, np.ndarray]):
            for id_, mask in masks.items():
                super().write(data_i, mask, appname=self.id_format(id_))

    def __init__(self, directory, clear = False) -> None:
        super().__init__(directory, clear)
        self.rgb_elements   = Elements(self, "rgb", 
                                       read_func=cv2.imread,  
                                       write_func=cv2.imwrite, 
                                       suffix='.jpg')

    def read_one(self, data_i):
        viewmeta = super().read_one(data_i)

        for k in viewmeta.bbox_3d:
            viewmeta.bbox_3d[k] = viewmeta.bbox_3d[k][:, ::-1]
        for k in viewmeta.landmarks:
            viewmeta.landmarks[k] = viewmeta.landmarks[k][:, ::-1]
        viewmeta.depth_scale *= 1000

        return viewmeta

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
