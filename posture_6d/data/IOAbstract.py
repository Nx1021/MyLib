# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_items, dict_keys, dict_values
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from open3d import geometry, utility, io
import sys
import os
import glob
import shutil
import pickle
import cv2
import time
from tqdm import tqdm
import types
import warnings

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Generator
from functools import partial

from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

DCT  = TypeVar('DCT',  bound="_DataCluster") # type of the cluster
DSNT = TypeVar('DSNT', bound='DatasetNode') # dataset node type
VDCT = TypeVar('VDCT') # type of the value of data cluster
VDST = TypeVar('VDST') # type of the value of dataset
from numpy import ndarray

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class _IOHook(Generic[DCT, VDCT], ABC):
    def __init__(self, iometa:"_IOMeta", io_func, hook_func = None) -> None:
        super().__init__()
        self.iometa:_IOMeta = iometa
        self.io_func = io_func
        self.hook_func = hook_func
        self._binded_kwargs:dict[str, Any] = {}

    def set_args(self, **kwargs):
        self._binded_kwargs = kwargs

    def clear_args(self):
        self._binded_kwargs = {}

    def __call__(self, key, *args, **kwargs) -> VDCT:
        rlt = self.io_func(key, *args, **kwargs, **self._binded_kwargs)
        self.hook_func(key, *args, **kwargs)
        return rlt

class _ReadHook(_IOHook, Generic[DCT, VDCT], ABC):
    def __init__(seMemorybonlf, iometa:"_IOMeta", 
                 io_func:Callable[[str, dict[str, Any]], VDCT], 
                 hook_func:Callable[[str, dict[str, Any]], None] = None) -> None:
        if hook_func is None:
            hook_func = lambda x, **kw: None
        super().__init__(iometa, io_func, hook_func)

    def __call__(self, key, **kwargs) -> VDCT:
        return super().__call__(key, **kwargs)

class _WriteHook(_IOHook, Generic[DCT, VDCT], ABC):
    def __init__(self, iometa:"_IOMeta", 
                 io_func:Callable[[str, VDCT, dict[str, Any]], None], 
                 hook_func:Callable[[str, VDCT, dict[str, Any]], None] = None) -> None:
        if hook_func is None:
            hook_func = lambda path, value, **kw: None
        super().__init__(iometa, io_func, hook_func)

    def __call__(self, key, value, **kwargs) -> VDCT:
        return super().__call__(key, value, **kwargs)

class _IOMeta(Generic[DCT, VDCT]):
    '''
    the position parameter of 'record', '_call' , '__call__' must be the same
    '''
    # IO_TYPE = WriteController.LOG_READ
    def __init__(self, cluster:DCT) -> None:
        super().__init__()
        self.cluster:DCT = cluster
        self.warning_info:str = "no description"

        self._kwargs = {}
        self.__once = False

    def record(self, **kwargs):
        pass

    def check_key(self, key, **kwargs) -> bool:
        return True
    
    def check_value(self, value, **kwargs) -> bool:
        return True

class _ReadMeta(_IOMeta[DCT, VDCT], ABC):
    def __init__(self, cluster: DCT, read_func = None) -> None:
        super().__init__(cluster)
        if read_func is None:
            read_func = self._call
        self.readhook = _ReadHook(self, read_func, self._hook)

    @abstractmethod
    def _call(self, key, **kwargs: Any) -> VDCT:
        pass

    def _hook(self, key, **kwargs):
        return

    def __call__(self, key, *args, force = False, **kwargs) -> VDCT:
        rlt = self.cluster._read_decorator(self)(key, *args, force = force, **kwargs)
        return rlt
    
    def record(self, key, *args, **kwargs):
        pass

class _WriteMeta(_IOMeta[DCT, VDCT], ABC):
    '''
    abstractmethod
    --
    * _call: the function to call when write
    * is_overwriting: check if the data is overwriting

    recommend to implement
    --
    * check_key: check if the key is valid
    * check_value: check if the value is valid
    '''
    @abstractmethod
    def _call(self, key, value, **kwargs: Any):
        pass
    
    @abstractmethod
    def is_overwriting(self, key, value, **kwargs):
        pass

    def __call__(self, key, value, *args, force = False, **kwargs) -> Any:
        rlt = self.cluster._write_decorator(self)(key, value, *args, force = force, **kwargs)
        return rlt    
    
    def record(self, key, value, **kwargs):
        pass

class InnerMemory(ABC, Generic[VDCT]):
    def __init__(self) -> None:
        self.memory_data:VDCT = None

    @abstractmethod
    def set(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def pop(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_empty(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass
    
class DictMemory(InnerMemory[dict[_KT, _VT]], Generic[_KT, _VT]):
    def __init__(self) -> None:
        super().__init__()
        self.memory_data:dict[_KT, _VT] = {}

    def set(self, key:_KT, value:_VT):
        self.memory_data[key] = value

class IOStatusWarning(Warning):
    pass


class IOStatusManager():
    def __init__(self) -> None:
        self.__closed = True
        self.__readonly = True
        self.__wait_writing = True
        self.__overwrite_allowed = False

    class _IOContext():
        def __init__(self, 
                     obj:"IOStatusManager", 
                     open = False, 
                     writing = False, 
                     overwrite_allowed = False) -> None:
            self.obj:IOStatusManager = obj
            self.orig_closed = None
            self.orig_readonly = None
            self.orig_overwrite_allowed = None
            self.open = open
            self.writing = writing
            self.overwrite_allowed = overwrite_allowed

        def __enter__(self):
            if self.open:
                self.orig_closed                = self.obj.closed
                self.orig_readonly              = self.obj.readonly
                self.orig_overwrite_allowed     = self.obj.overwrite_allowed
                self.obj.open()
                self.obj.set_writable(self.writing)
                self.obj.set_overwrite_allowed(self.overwrite_allowed)
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                if self.open:
                    self.obj.set_overwrite_allowed(self.orig_overwrite_allowed)
                    self.obj.set_readonly(self.orig_readonly)
                    self.obj.close(self.orig_closed)
                return True

    @property
    def closed(self):
        return self.__closed
    
    @property
    def opened(self):
        return not self.__closed

    @property
    def readonly(self):
        return self.__readonly and not self.__closed
    
    @property
    def writable(self):
        return not self.__readonly and not self.__closed

    @property
    def wait_writing(self):
        return self.__wait_writing and not self.__readonly and not self.__closed

    @property
    def is_writing(self):
        return not self.__wait_writing and not self.__readonly and not self.__closed

    @property
    def overwrite_allowed(self):
        return self.__overwrite_allowed and not self.__readonly and not self.__closed
    
    @property
    def overwrite_forbidden(self):
        return not self.__overwrite_allowed and not self.__readonly and not self.__closed
    
    def close(self, closed:bool = True):
        if not self.__closed and closed:
            self.set_wait_writing()
            self.set_readonly()
            self.close_hook()
        elif self.__closed and not closed:
            self.open_hook()
        self.__closed = closed

    def open(self, opened:bool = True):
        self.close(not opened)

    def set_readonly(self, readonly:bool = True):
        if self.closed:
            warnings.warn(f"the Status is closed, please call '{self.set_readonly.__name__}' when it's opened", IOStatusWarning)
        if not self.__readonly and readonly:
            self.set_wait_writing()
            self.readonly_hook()
        elif self.__readonly and not readonly:
            self.writable_hook()
        self.__readonly = readonly
        
    def set_writable(self, writable:bool = True):
        self.set_readonly(not writable)    

    def set_wait_writing(self, wait_writing:bool = True):
        if self.closed or self.readonly:
            warnings.warn(f"the Status is closed or readonly, please call '{self.set_wait_writing.__name__}' when it's writable", IOStatusWarning)
        if not self.__wait_writing and wait_writing:
            self.wait_writing_hook()
        elif self.__wait_writing and not wait_writing:
            self.is_writing_hook()
        self.__wait_writing = wait_writing

    def set_is_writing(self, is_writing:bool = True):
        self.set_wait_writing(not is_writing)

    def set_overwrite_allowed(self, overwrite_allowed:bool = True):
        if self.closed or self.readonly or self.wait_writing:
            warnings.warn(f"the Status is closed or readonly or writing, please call '{self.set_overwrite_allowed.__name__}' when it's writable", IOStatusWarning)
        self.__overwrite_allowed = overwrite_allowed

    def set_overwrite_forbidden(self, overwrite_forbidden:bool = True):
        self.set_overwrite_allowed(not overwrite_forbidden)

    ### IO operation ###
    def _read_decorator(self, iometa:_ReadMeta):
        '''
        brief
        -----
        Decorator function to handle reading operations when the cluster is closed. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        func = iometa._call
        warning_info:str = iometa.warning_info
        def wrapper(data_i, *args, force = False, **kwargs):
            nonlocal self, warning_info
            with self._Force(self, force, False): 
                read_error = False
                if self.is_closed(with_warning=True):
                    return None
                elif not iometa.check_key(data_i, **kwargs):
                    warning_info = f"key:{data_i} is not valid"
                    read_error = True
                
                try:
                    rlt = func(data_i, *args, **kwargs)  # Calls the original function.
                except ClusterDataIOError as e:
                    rlt = None
                    if str(e) == "skip":
                        pass
                    else:
                        read_error = True
                
                if read_error:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                                read {data_i} failed:\
                                    {warning_info}", ClusterIONotExecutedWarning)
            return rlt
        return wrapper

    def _write_decorator(self, iometa:_WriteMeta):
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
        func = iometa._call
        log_type:int = iometa.IO_TYPE
        warning_info:str = iometa.warning_info

        def wrapper(data_i, value = None, *args, force = False, **kwargs):
            nonlocal self, log_type, warning_info
            with self._Force(self, force): 
                write_error = False
                overwrited = False

                if self.is_closed(with_warning=True) or self.is_readonly(with_warning=True):
                    return None
                elif not iometa.check_key(data_i, **kwargs):
                    warning_info = f"key:{data_i} is not valid"
                    write_error = True
                elif not iometa.check_value(value, **kwargs):
                    warning_info = f"value:{value} is not valid"
                    write_error = True              
                elif iometa.is_overwriting(data_i, value, **kwargs):
                    if not self.overwrite_allowed and not force:
                        warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                                    is not allowed to overwitre, any write operation will not be executed.",
                                        ClusterIONotExecutedWarning)
                        write_error = True
                        return None
                    overwrited = True 

                if not write_error:                
                    try:
                        if not self.is_writing:
                            self.start_writing()
                        rlt = func(data_i, value, *args, **kwargs)  # Calls the original function.
                    except ClusterDataIOError as e:
                        rlt = None
                        if str(e) == "skip":
                            pass
                        else:
                            write_error = True     
                    else:
                        self.changed_since_opening = True  # Marks the cluster as updated after writing operations.
                        if overwrited and log_type == self.LOG_APPEND:
                            log_type = self.LOG_CHANGE
                        self._update_cluster_inc(iometa, data_i, value, *args, **kwargs)
                        self.log_to_mark_file(log_type, data_i, value)
                
                if write_error:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                        write {data_i}, {value} failed:\
                        {warning_info}", ClusterIONotExecutedWarning)

            return rlt
        return wrapper
    #####################


class A():
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass

class IOAbstractClass(ABC):
    def __init__(self) -> None:
        super().__init__()

