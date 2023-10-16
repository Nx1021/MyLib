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
import sys

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Generator
from functools import partial
import copy

from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt,\
      serialize_object, deserialize_object, read_file_as_str, write_str_to_file
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

DEBUG = False

IOSM = TypeVar('IOSM', bound="IOStatusManager") # type of the IO status manager

DMT  = TypeVar('DMT',  bound="DataMapping") # type of the cluster
DSNT = TypeVar('DSNT', bound='DatasetNode') # dataset node type
FCT = TypeVar('FCT', bound="FilesCluster") # type of the files cluster
FHT = TypeVar('FHT', bound="FilesHandle") # type of the files handle
VDMT = TypeVar('VDMT') # type of the value of data cluster
VDST = TypeVar('VDST') # type of the value of dataset
from numpy import ndarray

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class AmbiguousError(ValueError):
    pass

class IOMetaParameterError(ValueError):
    pass

class IOMetaPriorityError(ValueError):
    pass

class KeyNotFoundError(KeyError):
    pass

class ClusterDataIOError(RuntimeError):
    pass

class ClusterWarning(Warning):
    pass

class DataMapExistError(OSError):
    pass

class IOStatusWarning(Warning):
    pass

class ClusterIONotExecutedWarning(ClusterWarning):
    pass

class ClusterNotRecommendWarning(ClusterWarning):
    pass

class IOStatusManager():
    WRITING_MARK = '.writing'

    LOG_READ = 0
    LOG_ADD = 1
    LOG_REMOVE = 2
    LOG_CHANGE = 3
    LOG_MOVE   = 4
    LOG_OPERATION = 5
    LOG_KN = [LOG_READ, LOG_ADD, LOG_REMOVE, LOG_CHANGE, LOG_MOVE, LOG_OPERATION]

    _DEBUG = False

    def __init__(self, name) -> None:
        self.name = name

        self.__closed = True
        self.__readonly = True
        self.__wait_writing = True
        self.__overwrite_allowed = False

        self.__writer = self._Writer(self)

        self.__decorate_callable = False
        self.__decorate_next_callable = False
        self.__next_called_overwrite_allowed = False

    class _IOContext():
        DEFAULT_INPUT_OPEN                  = False
        DEFAULT_INPUT_WRITABLE              = False
        DEFAULT_INPUT_OVERWRITE_ALLOWED     = False
        def __init__(self, 
                     obj:"IOStatusManager") -> None:
            self.obj:IOStatusManager = obj
            
            self.orig_closed:bool               = True
            self.orig_readonly:bool             = True
            self.orig_overwrite_allowed:bool    = False

            self.reset_input()

            self.working = False

        def reset_input(self):
            self.input_open                 = self.DEFAULT_INPUT_OPEN                  
            self.input_writable             = self.DEFAULT_INPUT_WRITABLE              
            self.input_overwrite_allowed    = self.DEFAULT_INPUT_OVERWRITE_ALLOWED     

        def set_input(self, open = False, writable = False, overwrite_allowed = False):
            self.input_open                 = open
            self.input_writable             = writable
            self.input_overwrite_allowed    = overwrite_allowed
            return self

        def enter_hook(self):
            self.orig_closed                = self.obj.closed
            self.orig_readonly              = self.obj.readonly
            self.orig_overwrite_allowed     = self.obj.overwrite_allowed

            self.obj.open(self.input_open)
            self.obj.set_writable(self.input_writable)
            if self.input_writable:
                self.obj.start_writing()
            self.obj.set_overwrite_allowed(self.input_overwrite_allowed)

        def exit_hook(self):
            self.obj.set_overwrite_allowed(self.orig_overwrite_allowed)
            if self.obj.is_writing:
                self.obj.stop_writing()
            self.obj.set_readonly(self.orig_readonly)
            self.obj.close(self.orig_closed)
            return True   

        def __enter__(self):
            if self.working:
                raise RuntimeError(f"the IOContext of {self.obj.__class__}:: {self.obj.__class__.__name__}:{self.obj.name} is already working")
            
            self.working = True
            if IOStatusManager._DEBUG:
                print(f"enter:\t{self.obj.__class__}:: {self.obj.__class__.__name__}:{self.obj.name}")
            
            self.enter_hook()

            self.reset_input() # reset

            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                rlt = False
                if IOStatusManager._DEBUG:
                    print(f"exit:\t{self.obj.__class__}:: {self.obj.__class__.__name__}:{self.obj.name}")
                rlt = self.exit_hook()
                self.working = False
                return rlt

    class _Writer(_IOContext):
        DEFAULT_INPUT_OPEN                  = True
        DEFAULT_INPUT_WRITABLE              = True
        def allow_overwriting(self, overwrite_allowed = True):
            self.input_overwrite_allowed = overwrite_allowed
            return self

    class _Empty_Writer(_Writer):
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)

    def call_with_writer(self, valid, overwrite_allowed = False, only_next = True):
        if valid:
            self.__set_decorate_callable(not only_next, True, overwrite_allowed)
        return self
    
    def stop_call_with_writer(self):
        self.__reset_decorate_callable_paras()

    def __reset_decorate_callable_paras(self):
        self.__decorate_callable = False
        self.__decorate_next_callable = False
        self.__next_called_overwrite_allowed = False

    def __set_decorate_callable(self, decorate_callable, decorate_next_callable, next_called_overwrite_allowed):
        self.__decorate_callable = decorate_callable
        self.__decorate_next_callable = decorate_next_callable
        self.__next_called_overwrite_allowed = next_called_overwrite_allowed

    # def __getattribute__(self, name):
    #     rlt = super().__getattribute__(name)
    #     if self.__decorate_next_callable:
    #         if callable(rlt):
    #             rlt = self._write_decorator(rlt, force = self.__next_called_overwrite_allowed)
    #         if not self.__decorate_callable:
    #             self.__reset_decorate_callable_paras()
    #     return rlt

    @abstractmethod
    def get_writing_mark_file(self) -> str:
        pass

    def mark_exist(self):
        return os.path.exists(self.get_writing_mark_file())

    def remove_mark(self):
        if self.mark_exist():
            os.remove(self.get_writing_mark_file())

    def load_from_mark_file(self):
        file_path = self.get_writing_mark_file()
        if os.path.exists(file_path):
            result = []
            with open(file_path, 'r') as file:
                for line in file:
                    # 使用strip()函数移除行末尾的换行符，并使用split()函数分割列
                    columns = line.strip().split(', ')
                    assert len(columns) == 3, f"the format of {file_path} is wrong"
                    log_type, key, value_str = columns
                    # 尝试将第二列的值转换为整数
                    log_type = int(log_type)
                    assert log_type in self.LOG_KN, f"the format of {file_path} is wrong"
                    try: key = int(key)
                    except ValueError: pass
                    if value_str == 'None':
                        value = None
                    else:
                        try: value = int(value_str)
                        except: value = value_str
                    result.append([log_type, key, value])
            return result
        else:
            return None

    def log_to_mark_file(self, log_type, src=None, dst=None, value=None):
        ## TODO
        if src is None and dst is None and value is None:
            return 
        assert log_type in self.LOG_KN, f"log_type must be in {self.LOG_KN}"
        file_path = self.get_writing_mark_file()
        with open(file_path, 'a') as file:
            line = f"{log_type}, {src}, {dst}, {type(value)}\n"
            file.write(line)

    def get_writer(self, valid = True):
        ok = False
        if valid:
            ok = not self.is_writing
        
        if ok:
            return self.__writer
        else:
            return self._Empty_Writer(self) ###继续实现，修改之前的代码

    @property
    def closed(self):
        return self.__closed
    
    @property
    def opened(self):
        return not self.__closed

    @property
    def readonly(self):
        return not self.writable # self.__readonly and not self.closed
    
    @property
    def writable(self):
        return not self.__readonly and not self.closed

    @property
    def wait_writing(self):
        return not self.is_writing  #self.__wait_writing and not self.readonly and not self.closed

    @property
    def is_writing(self):
        return not self.__wait_writing and not self.readonly and not self.closed

    @property
    def overwrite_allowed(self):
        return not self.overwrite_forbidden # not self.__overwrite_allowed and not self.readonly and not self.closed
    
    @property
    def overwrite_forbidden(self):
        return not self.__overwrite_allowed and not self.readonly and not self.closed
    
    def close(self, closed:bool = True):
        if not self.closed and closed:
            self.stop_writing()
            self.set_readonly()
            self.close_hook()
        elif self.closed and not closed:
            self.open_hook()
        self.__closed = closed

    def open(self, opened:bool = True):
        self.close(not opened)

    def reopen(self):
        self.close()
        self.open()

    def set_readonly(self, readonly:bool = True):
        if (self.readonly ^ readonly) and self.closed:
            warnings.warn(f"the Status is closed, please call '{self.set_readonly.__name__}' when it's opened", IOStatusWarning)
        if not self.readonly and readonly:
            self.stop_writing()
            self.readonly_hook()
        elif self.readonly and not readonly:
            self.writable_hook()
        self.__readonly = readonly
        
    def set_writable(self, writable:bool = True):
        self.set_readonly(not writable)    

    def stop_writing(self, stop_writing:bool = True):
        if (self.wait_writing ^ stop_writing) and (self.closed or self.readonly):
            warnings.warn(f"the Status is closed or readonly, please call '{self.stop_writing.__name__}' when it's writable", IOStatusWarning)
        if not self.wait_writing and stop_writing:
            self.set_overwrite_forbidden()
            self.stop_writing_hook()
            if os.path.exists(self.get_writing_mark_file()):
                os.remove(self.get_writing_mark_file())
            self.__wait_writing = True
        elif self.wait_writing and not stop_writing:
            self.__wait_writing = False
            with open(self.get_writing_mark_file(), 'w'):
                pass
            self.start_writing_hook()

    def start_writing(self, start_writing:bool = True, overwrite_allowed:bool = False):
        self.set_overwrite_allowed(overwrite_allowed)
        self.stop_writing(not start_writing)

    def set_overwrite_allowed(self, overwrite_allowed:bool = True):
        if (self.overwrite_allowed ^ overwrite_allowed) and (self.closed or self.readonly):
            warnings.warn(f"the Status is closed or readonly, please call '{self.set_overwrite_allowed.__name__}' when it's writable", IOStatusWarning)
        if not self.overwrite_allowed and overwrite_allowed:
            self.set_overwrite_allowed_hook()
        elif self.overwrite_allowed and not overwrite_allowed:
            self.set_overwrite_forbidden_hook()
        self.__overwrite_allowed = overwrite_allowed

    def set_overwrite_forbidden(self, overwrite_forbidden:bool = True):
        self.set_overwrite_allowed(not overwrite_forbidden)
    
    def is_closed(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self.closed:
            warnings.warn(f"{self.__class__.__name__}:{self.name} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self.closed

    def is_readonly(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self.readonly:
            warnings.warn(f"{self.__class__.__name__}:{self.name} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self.readonly
    
    def close_hook(self):
        pass

    def open_hook(self):
        pass

    def readonly_hook(self):
        pass

    def writable_hook(self):
        pass

    def stop_writing_hook(self):
        pass

    def start_writing_hook(self):
        pass

    def set_overwrite_allowed_hook(self):
        pass

    def set_overwrite_forbidden_hook(self):
        pass
    
    @staticmethod
    def _write_decorator(func):
        def wrapper(self:IOStatusManager, *args, overwrite_allowed = False, **kwargs):
            with self.get_writer().allow_overwriting(overwrite_allowed):
                rlt = func(self, *args, **kwargs)
            return rlt
        return wrapper

class _Prefix(dict[str, str]):
    KW_PREFIX = "prefix"
    KW_JOINER = "joiner"
    def __init__(self, prefix:str = "", joiner = "") -> None:
        super().__init__()
        self[self.KW_PREFIX] = prefix
        self[self.KW_JOINER] = joiner

    @property
    def prefix(self):
        return self[self.KW_PREFIX]
    
    @prefix.setter
    def prefix(self, value):
        assert isinstance(value, str), f"prefix must be a str"
        self[self.KW_PREFIX] = value

    @property
    def joiner(self):
        return self[self.KW_JOINER]
    
    @joiner.setter
    def joiner(self, value):
        return self[self.KW_JOINER]
    
    def get_with_joiner(self):
        return self.prefix + self.joiner
    
    def __repr__(self) -> str:
        return f"Prefix({self.prefix}, {self.joiner})"
    
    def as_dict(self):
        return dict(self)
    
    @classmethod
    def from_dict(cls, dict_):
        prefix = dict_[cls.KW_PREFIX]
        joiner = dict_[cls.KW_JOINER]
        return cls(prefix, joiner)

class _AppendNames(dict[str, Union[list[str], str]]):
    KW_APPENDNAMES = "appendnames"
    KW_JOINER = "joiner"

    def __init__(self, appendnames:list[str] = None, joiner:str = '_') -> None:  # type: ignore
        super().__init__()   
        appendnames = appendnames if appendnames is not None else []
        self[self.KW_APPENDNAMES] = appendnames
        self[self.KW_JOINER] = joiner

    @property
    def joiner(self) -> str:
        return self[self.KW_JOINER] # type: ignore

    @property
    def appendnames(self) -> list[str]:
        return self[self.KW_APPENDNAMES] # type: ignore
 
    def get_with_joiner(self):
        rlt_list:list[str] = []
        for x in self.appendnames:
            x = self.joiner + x
            rlt_list.append(x)
        return rlt_list
    
    def extend(self, names):
        if isinstance(names, str):
            names = [names]
        assert isinstance(names, Iterable), f"names must be a iterable"
        assert all([isinstance(x, str) for x in names]), f"names must be a list of str"
        self.appendnames.clear()
        self.appendnames.extend(names)

    def add_appendname(self, appendname):
        if appendname not in self:
            self.appendnames.append(appendname)
            return True

    def remove_appendname(self, appendname):
        if appendname in self:
            self.appendnames.remove(appendname)
            return True
        else:
            return False

    @staticmethod
    def conditional_return(mutil_file, list_like:list[str]):
        if mutil_file:
            return list_like
        else:
            return list_like[0]

    def __repr__(self) -> str:
        return f"AppendNames({self})"
        
    @classmethod
    def from_dict(cls, dict_:dict):
        appendnames = dict_[cls.KW_APPENDNAMES]
        joiner = dict_[cls.KW_JOINER]
        return cls(appendnames, joiner)
    
    def as_dict(self):
        return dict(self)

class CacheProxy():
    KW_cache = "cache"
    KW_value_type = "value_type"

    def __init__(self, cache, value_type = None, value_init_func:Callable = None) -> None:
        self.__cache = None
        self.synced = False
        self.__value_type = value_type
        self.__value_init_func = value_init_func if value_init_func is not None else value_type

        self.cache = cache
        self.init_cache()
        
    @property
    def value_type(self) -> type:
        return self.__value_type # type: ignore
    
    # @value_type.setter
    # def value_type(self, value_type):
    #     assert isinstance(value_type, type), f"value_type must be a type"
    #     if value_type is not None:
    #         if self.cache is not None:
    #             assert type(self.cache) == value_type, f"the type of cache must be {value_type}, not {type(self.cache)}"
    #         else:
    #             self.__value_type = value_type
    #     else:
    #         self.__value_type = None

    @property
    def cache(self):
        return self.__cache
    
    @cache.setter
    def cache(self, cache):
        if self.value_type is not None and cache is not None:
            assert isinstance(cache, self.value_type), f"the type of cache must be {self.value_type}, not {type(cache)}"
        self.__cache = copy.deepcopy(cache)

    def as_dict(self):
        dict_ = {}
        dict_[self.KW_cache] = self.__cache
        dict_[self.KW_value_type] = self.__value_type
        return dict_
    
    @classmethod
    def from_dict(cls, dict_:dict):
        cache = dict_[cls.KW_cache]
        value_type = dict_[cls.KW_value_type]
        return cls(cache, value_type)
    
    def init_cache(self):
        if self.value_type is not None and self.cache is None:
            try:
                init_cache = self.__value_init_func()
            except Exception as e:
                if DEBUG:
                    raise Exception(e).with_traceback(sys.exc_info()[2])
                else:
                    pass
            else:
                self.cache = init_cache
                return True
        return False

class FilesHandle(Generic[FCT, VDMT]):
    '''
    immutable object, once created, it can't be changed.
    '''
    KW_data_path         = "data_path"
    KW_sub_dir           = "sub_dir"
    KW_corename          = "corename"
    KW_suffix            = "suffix"

    KW_appendnames       = "appendnames"   
    KW_prefix            = "prefix" 

    KW_read_func         = "read_func"
    KW_write_func        = "write_func"

    KW_cache             = "cache"  

    DEFAULT_FILE_TYPE = {
        ".json": [JsonIO.load_json, JsonIO.dump_json, dict],
        ".npy":  [partial(np.load, allow_pickle=True), partial(np.save, allow_pickle=True), None],
        ".npz":  [partial(np.load, allow_pickle=True), partial(np.savez, allow_pickle=True), None],
        ".pkl":  [deserialize_object, serialize_object, None],
        ".txt":  [read_file_as_str, write_str_to_file, None],
        ".png":  [cv2.imread, cv2.imwrite, None],
        ".jpg":  [cv2.imread, cv2.imwrite, None],
        ".jpeg": [cv2.imread, cv2.imwrite, None],
        ".bmp":  [cv2.imread, cv2.imwrite, None],
        ".tif":  [cv2.imread, cv2.imwrite, None],
    }

    DEFAULT_READ_FUNC = None
    DEFAULT_WRITE_FUNC = None
    DEFAULT_VALUE_TYPE = None
    DEFAULT_VALUE_INIT_FUNC = None

    LOAD_CACHE_ON_INIT = False

    KW_INIT_WITHOUT_CACHE = "INIT_WITHOUT_CACHE"

    _filehandles:dict[str, list["FilesHandle"]] = {}

    IGNORE_CREATE_SAME_PATH_FH_WARNING = False

    def __init__(self, cluster:FCT, sub_dir:str, corename:str, suffix:str, * ,
                 prefix:str = "", appendnames:list[str] = None,  # type: ignore
                 prefix_joiner:str = '', appendnames_joiner:str = '',
                 data_path = "",
                 read_func:Callable = None, write_func:Callable = None, 
                 cache = None, value_type:Callable = None) -> None: #type: ignore
        if hasattr(self, "_inited"):
            return
        super().__init__()
        self.cluster:FCT = cluster

        (   sub_dir, corename, suffix, 
            prefix, appendnames, prefix_joiner, appendnames_joiner, 
            data_path, 
            read_func, write_func, 
            cache, value_type) =\
            self.init_input_hook(sub_dir = sub_dir, corename = corename, suffix = suffix, 
                                prefix = prefix, appendnames = appendnames, 
                                prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                                data_path = data_path,
                                read_func = read_func, write_func = write_func,
                                cache = cache, value_type = value_type)

        appendnames = appendnames if appendnames is not None else ['']
        if len(suffix) == 0 or suffix[0] != '.':
            suffix = '.' + suffix
    
        data_path = self.cluster.data_path if data_path == "" else data_path
        assert data_path in cluster.data_path, f"data_path must be in {cluster.data_path}"

        self.data_path = data_path
        self.sub_dir = sub_dir
        self.corename = corename
        self.suffix = suffix

        self._prefix_obj         = _Prefix(prefix, prefix_joiner)
        self._appendnames_obj    = _AppendNames(appendnames, appendnames_joiner)

        self.read_func = read_func 
        self.write_func = write_func 

        if cache == self.KW_INIT_WITHOUT_CACHE:
            cache = None
        else:
            if self.LOAD_CACHE_ON_INIT and self.all_file_exist:
                cache = self.read()
            else:
                cache = cache

        self.cache_proxy:CacheProxy = CacheProxy(cache, value_type, self.DEFAULT_VALUE_INIT_FUNC)

        self.init_additional_hook()

        self._inited = True

        self._summary_ref()

    @classmethod
    def _ignore_create_same_path_fh_warning_once(cls):
        FilesHandle.IGNORE_CREATE_SAME_PATH_FH_WARNING = True
        return cls

    def _summary_ref(self):
        id_str = self.get_path()
        if id_str in self._filehandles:
            for obj in list(self._filehandles[id_str]):
                ref_count = sys.getrefcount(obj) - 1
                if ref_count == 1:
                    idx = [id(x) for x in self._filehandles[id_str]].index(id(obj))
                    self._filehandles[id_str].pop(idx)
            if len(self._filehandles[id_str]) > 0:
                if FilesHandle.IGNORE_CREATE_SAME_PATH_FH_WARNING:
                    FilesHandle.IGNORE_CREATE_SAME_PATH_FH_WARNING = False
                else:
                    warnings.warn(f"the FilesHandle:{id_str} has been created, please use the same FilesHandle", IOStatusWarning)
        self._filehandles.setdefault(id_str, []).append(self)

    def __hash__(self):
        return hash(self.get_path()) # + id(self)
    
    def __eq__(self, o: object) -> bool:
        if isinstance(o, FilesHandle):
            return hash(self) == hash(o)
        else:
            return False

    def immutable_attr_same_as(self, o: "FilesHandle") -> bool:
        rlt = self.data_path == o.data_path and \
                self.sub_dir == o.sub_dir and \
                self.corename == o.corename and \
                self.suffix == o.suffix and \
                self.prefix == o.prefix and \
                self.appendnames == o.appendnames and \
                self.read_func == o.read_func and \
                self.write_func == o.write_func and \
                self.value_type == o.value_type
        return rlt

    def __setattr__(self, name, value):
        if hasattr(self, "_inited"):
            raise AttributeError(f"FilesHandle is immutable, you can't change its attribute")
        return super().__setattr__(name, value)

    def init_input_hook(self, *, sub_dir, corename, suffix, 
                             prefix, appendnames, prefix_joiner, appendnames_joiner, 
                             data_path, 
                             read_func, write_func, 
                             cache, value_type):
        return (sub_dir, corename, suffix, 
                prefix, appendnames, prefix_joiner, appendnames_joiner, 
                data_path, 
                read_func, write_func, 
                cache, value_type)

    def init_additional_hook(self):
        pass

    ##IO##        
    @property
    def is_closed(self):
        return self.cluster.closed

    @property
    def is_readonly(self):
        return self.cluster.readonly

    @property
    def overwrite_forbidden(self):
        return self.cluster.overwrite_forbidden
    ######

    @property
    def multi_files(self):
        return self.cluster.MULTI_FILES

    @property
    def prefix(self) -> str:
        return self._prefix_obj.prefix

    @property
    def prefix_with_joiner(self) -> str:
        return self._prefix_obj.get_with_joiner()

    @property
    def appendnames(self) -> Union[list[str], str]:
        appendnames = self._appendnames_obj.appendnames
        return _AppendNames.conditional_return(self.multi_files, appendnames)

    @property
    def appendnames_with_joiner(self) -> Union[list[str], str]:
        apwj = self._appendnames_obj.get_with_joiner()
        return _AppendNames.conditional_return(self.multi_files, apwj)

    @property
    def synced(self):
        return self.cache_proxy.synced
    
    @property
    def value_type(self) -> type[VDMT]:
        return self.cache_proxy.value_type

    @property
    def full_directory(self):
        return os.path.join(self.data_path, self.sub_dir)

    @property
    def all_file_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return False
        return all([os.path.exists(x) for x in paths])
    
    @property
    def all_file_not_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return True
        return all([not os.path.exists(x) for x in paths])
    
    @property
    def any_file_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return False
        return any([os.path.exists(x) for x in paths])
    
    @property
    def any_file_not_exist(self):
        paths = self.get_path(get_list = True)
        if len(paths) == 0:
            return True
        return any([not os.path.exists(x) for x in paths])
    
    @property
    def has_cache(self):
        return (self.cache is not None)

    @property
    def empty(self):
        return not (self.any_file_exist or self.has_cache)

    @property
    def file_exist_status(self) -> list[bool]:
        paths = self.get_path(get_list=True)
        return [os.path.exists(x) for x in paths]

    @property
    def valid(self):
        return self.cluster is not None

    def get_name(self, get_list = False) -> Union[list[str], str]:
        if self.multi_files or get_list:
            awu_list:list[str] = self._appendnames_obj.get_with_joiner() 
            return [self.prefix_with_joiner + self.corename + x + self.suffix for x in awu_list]
        else:
            awu:str = self.appendnames_with_joiner # type: ignore
            return self.prefix_with_joiner + self.corename + awu + self.suffix 

    def get_dir(self):
        return os.path.join(self.data_path, self.sub_dir)

    def get_path(self, get_list = False) -> Union[list[str], str]:
        name = self.get_name(get_list = get_list)
        dir_ = self.get_dir()
        if isinstance(name, str):
            return os.path.join(dir_, name)
        else:
            return [os.path.join(dir_, x) for x in name]

    @property
    def cache(self) -> VDMT:
        # if not self.is_closed:
        #     if not self.is_readonly:
        #         return self.cache_proxy.cache
        #     else:
        #         return copy.copy(self.cache_proxy.cache)
        # else:
        #     return None
        if not self.is_readonly:
            return self.cache_proxy.cache
        else:
            return copy.copy(self.cache_proxy.cache)

    def set_cache(self, cache):
        if not self.is_closed and not self.is_readonly:
            if cache is None:
                self.erase_cache()
            else:
                self.cache_proxy.cache = cache

    def erase_cache(self):
        if not self.is_closed and not self.is_readonly:
            self.cache_proxy.cache = None    

    def _unsafe_get_cache(self) -> VDMT:
        '''
        not recommended, use with caution
        '''
        return self.cache_proxy.cache

    def read(self) -> Union[VDMT, list[VDMT]]:
        if self.read_func is not None:
            path = self.get_path()
            if isinstance(path, str):
                if os.path.exists(path):
                    return self.read_func(path)
            else:
                rlts = []
                for p in path:
                    if os.path.exists(p):
                        rlts.append(self.read_func(p))
                    else:
                        rlts.append(None)

    def set_synced(self, synced:bool = True):
        if self.has_cache and self.all_file_exist:
            pass
        elif not self.has_cache and self.all_file_not_exist:
            synced = True
        else:
            synced = False
        self.cache_proxy.synced = synced

    def as_dict(self):
        dict_ = {}
        dict_[self.KW_data_path]     = self.data_path
        dict_[self.KW_sub_dir]       = self.sub_dir
        dict_[self.KW_corename]      = self.corename
        dict_[self.KW_suffix]        = self.suffix
        dict_[self.KW_prefix]        = self._prefix_obj.as_dict()
        dict_[self.KW_appendnames]   = self._appendnames_obj.as_dict()
        dict_[self.KW_read_func]     = self.read_func
        dict_[self.KW_write_func]    = self.write_func
        dict_[self.KW_cache]         = self.cache_proxy.as_dict()
        return dict_

    @classmethod
    def from_dict(cls, cluster, dict_:dict):
        data_path   = dict_[cls.KW_data_path]
        sub_dir         = dict_[cls.KW_sub_dir]
        corename        = dict_[cls.KW_corename]
        suffix          = dict_[cls.KW_suffix]      

        prefix                  = dict_[cls.KW_prefix][_Prefix.KW_PREFIX]
        prefix_joiner           = dict_[cls.KW_prefix][_Prefix.KW_JOINER]
        appendnames             = dict_[cls.KW_appendnames][_AppendNames.KW_APPENDNAMES]
        appendnames_joiner      = dict_[cls.KW_appendnames][_AppendNames.KW_JOINER]

        read_func       = dict_[cls.KW_read_func]
        write_func      = dict_[cls.KW_write_func]

        cache           = dict_[cls.KW_cache][CacheProxy.KW_cache]
        value_type      = dict_[cls.KW_cache][CacheProxy.KW_value_type]

        obj = cls(cluster, sub_dir, corename, suffix,
            prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
            data_path = data_path,
            read_func = read_func, write_func = write_func,
            cache = cache, value_type = value_type)
        return obj
    
    @classmethod
    def from_path(cls, cluster:FCT, path:Union[str, list[str]], *,
                  prefix_joiner:str = '', appendnames_joiner:str = '', 
                  read_func:Callable = None, write_func:Callable = None, 
                  cache = None, value_type:Callable = None,  #type: ignore
                  _extract_corename_func:Callable[[str], tuple[str, str, str, str, str]] = None): #type: ignore
        datapath = cluster.data_path
        if isinstance(path, str):
            assert datapath in path, "cannot create a fileshandle object which is not in the data_path"
            filename = os.path.relpath(path, datapath)
        else:
            assert len(path) > 0, f"path must be a str or a list of str"
            assert all([datapath in p for p in path]), "cannot create a fileshandle object which is not in the data_path"
            filename = [os.path.relpath(p, datapath) for p in path]
        return cls.from_name(cluster, filename,
                                prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                                read_func = read_func, write_func = write_func, 
                                cache = cache, value_type = value_type,  #type: ignore
                                _extract_corename_func = _extract_corename_func)

    @classmethod
    def from_name(cls, cluster:FCT, filename:Union[str, list[str]], *,
                  prefix_joiner:str = '', appendnames_joiner:str = '', 
                  read_func:Callable = None, write_func:Callable = None, 
                  cache = None, value_type:Callable = None,  #type: ignore
                  _extract_corename_func:Callable[[str], tuple[str, str, str, str, str]] = None):
        def parse_one(filename:str):
            splitlist = filename.split(os.sep, 2)
            if len(splitlist) == 1:
                sub_dir, name = "", splitlist[0]
            else:
                sub_dir, name = splitlist[0], splitlist[1]
            basename, suffix = os.path.splitext(name)

            if _extract_corename_func is not None:
                corename, prefix, appendname, _prefix_joiner, _appendnames_joiner = _extract_corename_func(basename)
            else:
                _prefix_joiner = prefix_joiner
                _appendnames_joiner = appendnames_joiner
                if prefix_joiner == "":
                    prefix, rest = "", basename
                else:
                    prefix, rest = basename.split(prefix_joiner, 1)

                if appendnames_joiner == "":
                    corename, appendname = rest, ""
                else:
                    corename, appendname = rest.split(appendnames_joiner, 1)

            return sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner
        
        if isinstance(filename, str):
            sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner = parse_one(filename)
            appendnames = list[str]([appendname])
        else:
            assert len(filename) > 0, f"path must be a str or a list of str"
            sub_dirs, corenames, suffixes, prefixes, appendnames = [], [], [], [], []
            for n in filename:
                sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner = parse_one(n)
                sub_dirs.append(sub_dir)
                corenames.append(corename)
                suffixes.append(suffix)       
                prefixes.append(prefix)         
                appendnames.append(appendname)

            assert len(set(sub_dirs)) == 1, f"sub_dir must be the same"
            assert len(set(corenames)) == 1, f"corename must be the same"
            assert len(set(suffixes)) == 1, f"suffix must be the same"
            assert len(set(prefixes)) == 1, f"prefix must be the same"
            sub_dir = sub_dirs[0]
            corename = corenames[0]
            suffix = suffixes[0]
            prefix = prefixes[0]

        data_path = cluster.data_path
        return cls(cluster, sub_dir, corename, suffix,
                   prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                   data_path = data_path,
                   read_func = read_func, write_func = write_func, 
                   cache = cache, value_type = value_type)

    @classmethod
    def from_fileshandle(cls, cluster, file_handle:"FilesHandle", *,
                        sub_dir:str = None, corename:str = None, suffix:str = None, #type: ignore
                        prefix:str = None, appendnames:list[str] = None, prefix_joiner:str = None, appendnames_joiner:str = None, #type: ignore
                        read_func:Callable = None, write_func:Callable = None, 
                        cache = None, value_type:Callable = None): #type: ignore
        sub_dir = file_handle.sub_dir if sub_dir is None else sub_dir
        corename = file_handle.corename if corename is None else corename
        suffix = file_handle.suffix if suffix is None else suffix

        prefix = file_handle.prefix if prefix is None else prefix
        appendnames = file_handle._appendnames_obj.appendnames if appendnames is None else appendnames
        prefix_joiner = file_handle._prefix_obj.joiner if prefix_joiner is None else prefix_joiner
        appendnames_joiner = file_handle._appendnames_obj.joiner if appendnames_joiner is None else appendnames_joiner

        read_func = file_handle.read_func if read_func is None else read_func
        write_func = file_handle.write_func if write_func is None else write_func

        cache = file_handle.cache if cache is None else cache
        value_type = file_handle.value_type if value_type is None else value_type
        
        return cls(cluster, sub_dir, corename, suffix,
                    prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
                    data_path = "",
                    read_func = read_func, write_func = write_func, 
                    cache = cache, value_type = value_type)

    @classmethod
    def create_not_exist_fileshandle(cls, cluster:FCT):
        return cls(cluster, "", "notexist", ".fhnotexist")
                                     
    def clear_notfound(self):
        new_appendnames = []
        for i, e in enumerate(self.file_exist_status):
            if e:
                new_appendnames.append(self._appendnames_obj.appendnames[i])
        return self.from_fileshandle(self.cluster, self, appendnames = new_appendnames)

    @classmethod
    def try_get_default(cls, file:str,
                            read_func:Callable = None, 
                            write_func:Callable = None, 
                            value_type:Callable = None):
        def get_with_priority(parameter, cls_default, public_default):
            if parameter is not None:
                return parameter
            elif cls_default is not None:
                return cls_default
            else:
                return public_default
        
        if file.startswith('.'):
            suffix = file
        else:
            suffix = os.path.splitext(file)[1]
        if suffix in cls.DEFAULT_FILE_TYPE:
            _read_func, _write_func, _value_type = cls.DEFAULT_FILE_TYPE[suffix]
            read_func = get_with_priority(read_func, cls.DEFAULT_READ_FUNC, _read_func)
            write_func = get_with_priority(write_func, cls.DEFAULT_WRITE_FUNC, _write_func)
            value_type = get_with_priority(value_type, cls.DEFAULT_VALUE_TYPE, _value_type)
        else:
            warnings.warn(f"can't find default file type for {suffix}, use str as default value_type", ClusterNotRecommendWarning)

        return read_func, write_func, value_type

    def get_key(self):
        return self.cluster.MemoryData._reverse_dict[self] # type: ignore

    def __repr__(self) -> str:
        if len(self._appendnames_obj.appendnames) == 1:
            string = f"FilesHandle({self.get_path()})"
        else:
            paths = self.get_path(get_list=True)
            string = f"FilesHandle({paths[0]}) + {len(paths) - 1} files"
        string += f" file exist: {self.file_exist_status}, has cache: {self.has_cache}, synced: {self.synced}"
        return string

class BinDict(dict[_KT, _VT], Generic[_KT, _VT]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_reverse_dict()

    def init_reverse_dict(self):
        self._reverse_dict = {value:key for key, value in self.items()}

    def __getitem__(self, __key: _KT) -> _VT:
        return super().__getitem__(__key)

    def __del_item(self, key):
        value = self[key]
        super().__delitem__(key)
        if value in self._reverse_dict:
            del self._reverse_dict[value]
        return value

    def __setitem__(self, key, value):
        if key in self:
            # 如果key已存在，则先删除旧的反向映射
            old_value = self[key]
            if old_value in self._reverse_dict:
                del self._reverse_dict[old_value]
        if value in self._reverse_dict:
            # 如果value已存在于反向映射中，则删除旧的key
            old_key = self._reverse_dict[value]
            del self[old_key]
        super().__setitem__(key, value)
        self._reverse_dict[value] = key

    def update(self, *args:dict, **kwargs):
        for other_dict in args:
            for key, value in other_dict.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key, default=None):
        if key in self:
            value = self.__del_item(key)
            return value
        else:
            return default

    def popitem(self):
        key, value = super().popitem()
        if value in self._reverse_dict:
            del self._reverse_dict[value]
        return key, value

    def clear(self):
        super().clear()
        self._reverse_dict.clear()

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __delitem__(self, key):
        if key in self:
            self.__del_item(key)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

class DataMapping(IOStatusManager, ABC, Generic[_VT, DMT, VDMT]):
    _registry:dict = {}

    _DMT = TypeVar('_DMT', bound="DataMapping")
    _VDMT = TypeVar('_VDMT')

    MULTI_FILES = False
    MEMORY_DATA_FILE = ".datamap"
    MEMORY_DATA_TYPE = BinDict

    KEY_TYPE = int
    FILESHANDLE_TYPE:type[FilesHandle] = FilesHandle

    load_memory_func:Callable[[str], dict]         = deserialize_object
    save_memory_func:Callable[[str, dict], None]   = serialize_object

    #############
    def __new__(cls, top_directory:str, name: str = "", *args, **kwargs):
        if cls is DataMapping:
            raise TypeError("DataMapping cannot be instantiated")
        # single instance
        obj = super().__new__(cls)
        obj.init_identity(top_directory, name, *args, **kwargs)
        if obj.identity_string() in cls._registry:
            obj = cls._registry[obj.identity_string()]
        else:
            cls._registry[obj.identity_string()] = obj
        return obj

    @staticmethod
    def parse_identity_string(identity_string:str):
        cls_name, directory_name = identity_string.split(':')
        directory, mapping_name = os.path.split(directory_name)
        return cls_name, directory, mapping_name

    @classmethod
    def gen_identity_string(cls, data_path):
        return f"{cls.__name__}:{data_path}"

    def identity_string(self):
        return self.gen_identity_string(self.data_path)

    def init_identity(self, top_directory:str, name: str, *args, **kwargs):
        self._unfinished_operation = 0
        self._top_directory = top_directory
        self.mapping_name = name        

    def key_identity_string(self):
        return f"{self.__class__.__name__}:{self.name}"        
    #############

    # def __init_subclass__(cls, **kwargs):
    #     write_func_name = [DataMapping.write.__name__, 
    #                        DataMapping.modify_key.__name__, 
    #                        DataMapping.remove.__name__, 
    #                        DataMapping.merge_from.__name__, 
    #                        DataMapping.append.__name__,
    #                        DataMapping.clear.__name__,
    #                        DataMapping.make_continuous.__name__]
    #     for name in write_func_name:
    #         sub_cls_func:Callable = getattr(cls, name)
    #         if sub_cls_func.__name__ == name:
    #             setattr(cls, name, IOStatusManager.force_decorator(sub_cls_func))
    #     super().__init_subclass__(**kwargs)

    def __init_subclass__(cls, **kwargs):
        ### __init__ ###
        cls.__init__ = cls.method_exit_hook_decorator(cls.__init__, cls.try_open)
        ### clear ###
        cls.clear = cls.method_exit_hook_decorator(cls.clear, cls._clear_empty_dir)
        super().__init_subclass__(**kwargs)

    def __init__(self, top_directory:Union[str, "DatasetNode"], name: str, *args, **kwargs) -> None:
        '''
        Initialize the data cluster with the provided top_directory, name, and registration flag.
        '''
        IOStatusManager.__init__(self, name)
        Generic.__init__(self)
        self._MemoryData:dict[int, _VT] = self.load_postprocess({})

        self.cache_priority     = True
        self.strict_priority_mode    = False
        self.write_synchronous  = False
        self.changed_since_opening = False

    @classmethod
    def method_exit_hook_decorator(cls, func, hook_func):
        def wrapper(self:DataMapping, *args, **kw):
            func(self, *args, **kw)
            if cls == self.__class__:
                hook_func(self)
        return wrapper

    def try_open(self):
        if os.path.exists(self.data_path):
            self.open()  # Opens the cluster for operation.
        else:
            self.close()

    def _clear_empty_dir(self):
        for r, d, f in os.walk(self.data_path):
            for dir_ in d:
                if len(os.listdir(os.path.join(r, dir_))) == 0:
                    os.rmdir(os.path.join(r, dir_))
    @property
    def top_directory(self):
        return self._top_directory

    @top_directory.setter
    def top_directory(self, value):
        return None

    def make_path(self):
        if not os.path.exists(self.data_path):
            if '.' in os.path.basename(self.data_path):
                dir_ = os.path.dirname(self.data_path)
                os.makedirs(dir_, exist_ok=True)
                with open(self.data_path, 'w'):
                    pass
            else:
                os.makedirs(self.data_path)

    def open_hook(self):
        self.make_path()
        self.load()
        self.reset_changed()
        
    def stop_writing_hook(self):
        self.sort()
        self.save()

    def get_writing_mark_file(self):
        return os.path.join(self.top_directory, self.mapping_name, self.WRITING_MARK)
    
    def set_changed(self):
        self.changed_since_opening = True

    def reset_changed(self):
        self.changed_since_opening = False
    
    @property
    def MemoryData(self):
        return self._MemoryData
    
    @property
    def MemoryData_path(self):
        return os.path.join(self.top_directory, self.mapping_name, self.MEMORY_DATA_FILE)
    
    @property
    def data_path(self):
        return os.path.join(self.top_directory, self.mapping_name).replace('\\', '/')

    def save_preprecess(self):
        return self.MemoryData

    def load_postprocess(self, data):
        return data

    def remove_memory_data_file(self):
        if os.path.exists(self.MemoryData_path):
            os.remove(self.MemoryData_path)

    def save(self):
        self.__class__.save_memory_func(self.MemoryData_path, self.save_preprecess())

    def load(self):
        if os.path.exists(self.MemoryData_path):
            MemoryData = self.load_postprocess(self.__class__.load_memory_func(self.MemoryData_path))
            self.merge_MemoryData(MemoryData) ## TODO
        else:
            if len(self.MemoryData) > 0:
                if DEBUG:
                    warnings.warn(f"will not rebuild")
            else:
                self.rebuild()
            self.save()
        if DEBUG:
            print(self)
            print(self.MemoryData)

    @abstractmethod
    def merge_MemoryData(self, MemoryData:dict):
        pass

    def sort(self):
        new_dict = dict(sorted(self.MemoryData.items(), key=lambda x:x[0])) # type: ignore
        self.MemoryData.clear()
        self.MemoryData.update(new_dict)    

    @property
    def num(self):
        return len(self.keys())
    
    @property
    def i_upper(self):
        return max(self.keys(), default=-1) + 1

    @property
    def continuous(self):
        return self.num == self.i_upper

    @abstractmethod
    def rebuild(self):
        pass

    ####################
    @abstractmethod
    def read(self, src:int) -> VDMT:
        pass

    @abstractmethod
    def write(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        pass
        
    @abstractmethod
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        pass

    @abstractmethod
    def remove(self, dst:int, *, force = False, **other_paras) -> None:
        pass

    @abstractmethod
    def merge_from(self, src:DMT, *, force = False) -> None:
        pass

    @abstractmethod
    def copy_from(self, src:DMT, *, cover = False, force = False) -> None:
        pass
    ####################

    #######################

    def keys(self):
        return self.MemoryData.keys()

    def values(self) -> Generator[VDMT, Any, None]:
        def value_generator():
            for i in self.keys():
                yield self.read(i)
        return value_generator()
    
    def items(self):
        def items_generator():
            for i in self.keys():
                yield i, self.read(i)
        return items_generator()

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
                    yield self.read(i)
            return g()
        elif isinstance(data_i, int):
            # 处理单个索引操作
            return self.read(data_i)
        else:
            raise TypeError("Unsupported data_i type")
    
    def __setitem__(self, data_i, value:VDMT):
        return self.write(data_i, value)

    def __iter__(self) -> Iterable[VDMT]:
        return self.values()
    
    def __len__(self):
        return len(self.MemoryData)
    ####################

    ### complex io ####
    def append(self, value:VDMT, *, force = False, **other_paras):
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        dst = self.i_upper
        self.write(dst, value, force=force, **other_paras)

    def clear(self, *, force = False):
        with self.get_writer(force).allow_overwriting():
            ### TODO
            for key in tqdm(list(self.keys()), desc=f"clear {self}"):
                self.remove(key)

    def make_continuous(self, *, force = False):
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        if self.continuous:
            return
        with self.get_writer(force).allow_overwriting():
            for i, key in tqdm(enumerate(list(self.keys()))):
                self.modify_key(key, i)

    ##########
    # def choose_unfinished_operation(obj):
    #     '''
    #         0. skip
    #         1. clear the unfinished data
    #         2. try to rollback the unfinished data
    #         3. exit"))
    #     '''
    #     choice = int(input(f"please choose an operation to continue:\n\
    #                 0. \n\
    #                 1. clear the unfinished data\n\
    #                 2. try to rollback the unfinished data\n\
    #                 3. ignore\n"))
    #     if choice not in [0, 1, 2, 3]:
    #         raise ValueError(f"invalid choice {choice}")
    #     return choice

    ### def process_unfinished(self):
    #     
    #     if self._unfinished:
    #         if self._unfinished_operation == 0:
    #             self._unfinished_operation = self.choose_unfinished_operation()
    #         if self._unfinished_operation == 0:
    #             return False
    #         self.rebuild()
    #         if self._unfinished_operation == 1:
    #             self.clear(force=True, ignore_warning=True)
    #             self._unfinished = False
    #             self.remove_mark()
    #             self.rebuild()
    #             return True
    #         elif self._unfinished_operation == 2:
    #             # try roll back
    #             log = self.load_from_mark_file()
    #             with self.writer.allow_overwriting():
    #                 for log_i, data_i, _ in log:
    #                     if log_i == self.LOG_APPEND:
    #                         if data_i in self.keys():
    #                             self.remove(data_i)
    #                             print(f"try to rollback, {data_i} in {self.identity_string()} is removed.")
    #                     else:
    #                         raise ValueError("can not rollback")
    #             self.remove_mark()
    #             return True
    #         elif self._unfinished_operation == 3:
    #             # reinit
    #             self.remove_mark()
    #             self.rebuild()
    #         else:
    #             raise ValueError(f"invalid operation {self._unfinished_operation}")
    #     if self._unfinished_operation == 3:
    #         # reinit
    #         self.rebuild()
    ##########

    def __repr__(self):
        return self.identity_string()
    
    @classmethod
    def _test_io(cls, d0:DMT, d1:DMT):
        pass

    @classmethod
    def _test(cls):
        clsname = f"{cls.__name__}"

        top_dir = os.path.join(os.getcwd(), f"{clsname}Test")

        d0 = cls(top_dir, f"{clsname}0")
        d1 = cls(top_dir, f"{clsname}1")

        d0.clear(force=True)
        d1.clear(force=True)

        cls._test_io(d0, d1)

        d0.rebuild()

        start = time.time()
        with d1.get_writer():
            for array in d0: # type: ignore
                d1.append(array)
        print("append passed", time.time() - start)

        start = time.time()
        d1.clear(force=True)
        print("clear passed", time.time() - start)

        start = time.time()
        with d1.get_writer():
            d1.copy_from(d0, cover=True) # type: ignore
        print("copy_from passed", time.time() - start)

        with d1.get_writer():
            d1.remove(0)
        start = time.time()
        d1.make_continuous(force=True)
        print("make_continuous passed", time.time() - start)

        return d0, d1

class IOMeta(ABC, Generic[FCT, VDMT, FHT]):
    '''
        src_handle, dst_handle = self.get_FilesHandle(src = src, dst = dst, **other_paras)
        preprocessed_value = self.preprogress_value(value, **other_paras)
        rlt = self.io(src_handle, dst_handle, preprocessed_value)
                  - preprocessed_value = self.format_value(preprocessed_value)
                    ...
                    rlt = self.inv_format_value(rlt)
        postprocessed_value = self.postprogress_value(rlt, **other_paras)
        self.progress_FilesHandle(src_handle, dst_handle, **other_paras)
    '''
    READ = True
    PATH_EXISTS_REQUIRED = True
    LOG_TYPE = IOStatusManager.LOG_READ
    WARNING_INFO = "no description"

    W_SYNC = False

    OPER_ELEM = False

    def __init__(self, files_cluster:FCT) -> None:
        self.files_cluster:FCT = files_cluster
        self.core_func:Callable = None #type: ignore
        self.core_func_binded_paras = {}

        self.__cache_priority = True
        self.__strict_priority_mode = False
        self.__write_synchronous = False
        self.__io_raw = False

        self.ctrl_mode = False

        self.save_memory_after_writing = False

    def _set_ctrl_flag(self, cache_priority=True, strict_priority_mode=False, write_synchronous=False, io_raw = False):
        self.__cache_priority = cache_priority
        self.__strict_priority_mode = strict_priority_mode
        self.__write_synchronous = write_synchronous
        self.__io_raw = io_raw
        self.ctrl_mode = True
        return self
    
    def _clear_ctrl_flag(self):
        self.__cache_priority = True
        self.__strict_priority_mode = False
        self.__write_synchronous = False
        self.__io_raw = False
        self.ctrl_mode = False
    
    def __enter__(self):
        self.ctrl_mode = True
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)
        self.ctrl_mode = False

    def _set_synced_flag(self, src_handle:FilesHandle, dst_handle:FilesHandle, synced = False):
        if src_handle is not None:
            src_handle.set_synced(synced)
        if dst_handle is not None:
            dst_handle.set_synced(synced)

    @property
    def key_type(self):
        return self.files_cluster.KEY_TYPE

    @property
    def multi_files(self):
        return self.files_cluster.MULTI_FILES

    @property
    def cache_priority(self):
        if self.ctrl_mode:
            return self.__cache_priority
        else:
            return self.files_cluster.cache_priority

    @property
    def strict_priority_mode(self):
        if self.ctrl_mode:
            return self.__strict_priority_mode
        else:
            return self.files_cluster.strict_priority_mode

    @property
    def write_synchronous(self):
        if self.ctrl_mode:
            return self.__write_synchronous
        else:
            return self.files_cluster.write_synchronous

    @property
    def io_raw(self):
        if self.ctrl_mode:
            return self.__io_raw
        else:
            return False

    @property
    def _FCMemoryData(self)  -> BinDict[int, FHT]:
        return self.files_cluster.MemoryData # type: ignore

    @abstractmethod
    def get_FilesHandle(self, src, dst, value, **other_paras) -> tuple[FHT, FHT]:
        pass

    def _query_fileshandle(self, data_i:int) -> FHT:
        return self.files_cluster.query_fileshandle(data_i)

    def get_file_core_func(self, src_file_handle:FHT, dst_file_handle:FHT, value) -> Callable:
        return None # type: ignore

    def progress_FilesHandle(self, 
                            src_file_handle:FHT, 
                            dst_file_handle:FHT, 
                            postprocessed_value, 
                            **other_paras) -> tuple[FHT]: # type: ignore
        pass

    @abstractmethod
    def io_cache(self, src_file_handle:FHT, dst_file_handle:FHT, value = None) -> Any:
        pass

    @abstractmethod
    def cvt_to_core_paras(self, src_file_handle:FHT, dst_file_handle:FHT, value) -> tuple:
        pass

    def preprogress_value(self, value, **other_paras) -> Any:
        return value

    def postprogress_value(self, value, **other_paras) -> Any:
        return value

    def format_value(self, value:VDMT) -> Any:
        return value
    
    def inv_format_value(self, formatted_value) -> VDMT:
        return formatted_value

    def core_func_hook(self, *core_args):
        pass

    def io_file(self, src_file_handle, dst_file_handle, value = None) -> Any:
        value = self.format_value(value)
        path, *core_values = self.cvt_to_core_paras(src_file_handle = src_file_handle, 
                                           dst_file_handle = dst_file_handle, 
                                           value = value)
        if not os.path.exists(path) and self.PATH_EXISTS_REQUIRED:
            raise IOMetaPriorityError("file not found")
        core_func = self.get_file_core_func(src_file_handle, dst_file_handle, value)
        core_func = self.core_func if core_func is None else core_func
        if core_func is not None:
            if self.multi_files:
                core_values = self.split_value_as_mutil(*core_values)
            rlt = self.execute_core_func(core_func, path, *core_values)
            if self.multi_files:
                return self.gather_mutil_results(rlt)
        else:
            raise IOMetaPriorityError("core_func is None")
        rlt = self.inv_format_value(rlt)
        return rlt

    def io(self, src_handle:FilesHandle, dst_handle:FilesHandle, preprocessed_value):
        priority_ok = False
        secondary_ok = False
        if self.cache_priority:
            func_priority = self.io_cache
            func_secondary = self.io_file
        else:
            func_priority = self.io_file
            func_secondary = self.io_cache
        try:
            rlt = func_priority(src_handle, dst_handle, preprocessed_value) # type: ignore
            priority_ok = True
            exe_secondary = not self.READ and (self.write_synchronous or self.W_SYNC)
        except IOMetaPriorityError:
            if self.strict_priority_mode:
                raise ClusterDataIOError
            exe_secondary = True
        
        if exe_secondary:
            try:
                rlt = func_secondary(src_handle, dst_handle, preprocessed_value) # type: ignore
            except IOMetaPriorityError:
                if priority_ok == False:
                    raise ClusterDataIOError
            else:
                secondary_ok = True

        if priority_ok and secondary_ok:
            self._set_synced_flag(src_handle, dst_handle, True)

        return rlt # type: ignore

    def operate_elem(self, src, dst, value, **other_paras):
        raise NotImplementedError

    def __call__(self, *, src = None, dst = None, value = None, **other_paras) -> Any:
        if not self.__io_raw:
            value = self.preprogress_value(value, **other_paras)
        if self.OPER_ELEM:
            rlt = self.operate_elem(src, dst, value, **other_paras)
        else:
            src_handle, dst_handle = self.get_FilesHandle(src = src, dst = dst, value = value, **other_paras)
            rlt = self.io(src_handle, dst_handle, value)
            self.progress_FilesHandle(src_handle, dst_handle, rlt, **other_paras)
        if not self.__io_raw:
            rlt = self.postprogress_value(rlt, **other_paras)
        return rlt

    def gather_mutil_results(self, results:list):
        raise NotImplementedError

    def split_value_as_mutil(self, *core_values):
        raise NotImplementedError

    def check_src(self, src):
        if not isinstance(src, self.key_type):
            return False
        if not self.files_cluster.has(src):
            return False
        return True
        
    def check_dst(self, dst):
        if not isinstance(dst, self.key_type):
            return False
        return True

    def check_value(self, value: Any):
        return True

    def is_overwriting(self, dst:int):
        return not self.files_cluster.idx_unwrited(dst)

    def execute_core_func(self, core_func, *core_args, **other_paras):
        if self.multi_files:
            rlts = []
            for i in range(len(core_args[0])):
                rlt = core_func(*[x[i] for x in core_args], **self.core_func_binded_paras)
                self.core_func_hook(*[x[i] for x in core_args], **other_paras)
                rlts.append(rlt)
            return rlts
        else:
            rlt = core_func(*core_args, **self.core_func_binded_paras)
            self.core_func_hook(*core_args)
            return rlt

class FilesCluster(DataMapping[FHT, FCT, VDMT], Generic[FHT, FCT, DSNT, VDMT]):

    _IS_ELEM = False
    _ELEM_BY_CACHE = False
    KEY_TYPE = int
    
    ALWAYS_ALLOW_WRITE = False
    ALWAYS_ALLOW_OVERWRITE = False

    #############
    def init_identity(self, dataset_node:Union[str, "DatasetNode"], name: str, *args, **kwargs):
        if isinstance(dataset_node, str):
            self._unfinished_operation = 0
            self._dataset_node:DSNT = None # type: ignore 
            self._top_directory = dataset_node
        elif isinstance(dataset_node, DatasetNode):
            self._unfinished_operation = dataset_node._unfinished_operation
            self._dataset_node:DSNT = dataset_node # type: ignore
        elif dataset_node is None:
            self._unfinished_operation = 0
            self._dataset_node:DSNT = None
            self._top_directory = ""
        else:
            raise TypeError(f"dataset_node must be str or DatasetNode, not {type(dataset_node)}")
        self.mapping_name = name        
    #############

    def __init__(self, dataset_node: Union[str, "DatasetNode"], name: str, *args, **kwargs) -> None:
        super().__init__(dataset_node, name)

        self._unfinished = self.mark_exist()
        self._unfinished_operation = 0

        self.init_io_metas()
        # self.register_to_dataset()

    @property
    def MemoryData(self) -> BinDict[int, FHT]:
        return self._MemoryData

    @property
    def top_directory(self) -> str:
        if self._dataset_node is not None:
            return self._dataset_node.top_directory
        else:
            return self._top_directory

    @top_directory.setter
    def top_directory(self, value):
        return None

    @property
    def dataset_node(self):
        return self._dataset_node
    
    @dataset_node.setter
    def dataset_node(self, dataset_node):
        if self._dataset_node is not None:
            self.unregister_from_dataset()
        self._dataset_node = dataset_node
        self.register_to_dataset()

    @property
    def registerd(self):
        return self._dataset_node is not None and \
            self.identity_string() in self._dataset_node.clusters_map

    def register_to_dataset(self):
        if self._dataset_node is not None:
            self._dataset_node.add_cluster(self)

    def unregister_from_dataset(self):
        if self.identity_string() in self._dataset_node.clusters_map:
            self._dataset_node.remove_cluster(self)

    ####### fileshandle operation ########
    def query_fileshandle(self, data_i:int, stop = None, step = None) -> Union[Generator[FHT, Any, None], FHT]:
        if isinstance(stop, (int, str)):
            stop = len(self) if stop == "end" else stop
            step = 1 if step is None else step
            assert isinstance(stop, int), f"stop must be int, not {type(stop)}"
            assert isinstance(step, int), f"step must be int, not {type(step)}"
            # return a generator
            def g():
                for i in range(data_i, stop, step):
                    yield self.MemoryData[i]
            return g()
        else:
            return self.MemoryData[data_i]
    
    @abstractmethod
    def create_fileshandle(self, src, dst, value, **other_paras) -> FHT:
        pass

    def format_corename(self, data_i:int):
        return None
    
    def deformat_corename(self, corename:str):
        return None
    
    def _set_fileshandle(self, data_i, fileshandle:FHT):
        # if self.closed:
        #     raise ClusterDataIOError("can not set fileshandle when cluster is closed")
        if fileshandle not in self.MemoryData._reverse_dict:
            self.MemoryData[data_i] = fileshandle
            return True
        return False

    def _pop_fileshandle(self, data_i):
        # if self.closed:
        #     raise ClusterDataIOError("can not set fileshandle when cluster is closed")
        if self.has_data(data_i):
            return self.MemoryData.pop(data_i)
    ##################

    #### io metas #####
    def init_io_metas(self):
        self.read_meta:IOMeta[FCT, VDMT, FHT]            = self._read(self)
        self.write_meta:IOMeta[FCT, VDMT, FHT]           = self._write(self)
        self.modify_key_meta:IOMeta[FCT, VDMT, FHT]      = self._modify_key(self)
        self.remove_meta:IOMeta[FCT, VDMT, FHT]          = self._remove(self)
        self.paste_file_meta:IOMeta[FCT, VDMT, FHT]      = self._paste_file(self)
        self.change_dir_meta:IOMeta[FCT, VDMT, FHT]      = self._change_dir(self)

    def cvt_key(self, key):
        return key

    def io_decorator(self, io_meta:IOMeta, force = False):
        func = io_meta.__call__
        is_read = io_meta.READ
        log_type = io_meta.LOG_TYPE
        warning_info = io_meta.WARNING_INFO

        allow_write         = force or self.ALWAYS_ALLOW_WRITE
        allow_overwrite     = force or self.ALWAYS_ALLOW_OVERWRITE
        
        def wrapper(*, src:int = None, dst:int = None, value = None, **other_paras): # type: ignore
            nonlocal self, log_type, warning_info
            src = self.cvt_key(src)
            dst = self.cvt_key(dst)
            rlt = None
            # with self._IOContext(self, force, force, force): 
            with self.get_writer(allow_write).allow_overwriting(allow_overwrite):
                io_error = False
                overwrited = False

                if self.is_closed(with_warning=True) or (not is_read and self.is_readonly(with_warning=True)):
                    return None
                if src is not None and not io_meta.check_src(src):
                    io_error = True
                elif dst is not None and not io_meta.check_dst(dst):
                    io_error = True
                elif value is not None and not io_meta.check_value(value):
                    io_error = True
                if io_meta.is_overwriting(dst):
                    if not self.overwrite_allowed and not force:
                        warnings.warn(f"{self.__class__.__name__}:{self.mapping_name} " + \
                                    "is not allowed to overwitre, any write operation will not be executed.",
                                    ClusterIONotExecutedWarning)
                        io_error = True
                        return None
                    overwrited = True 

                if not io_error:                
                    try:
                        if not is_read and not self.is_writing:
                            self.start_writing()
                        rlt = func(src=src, dst = dst, value = value, **other_paras)  # Calls the original function.
                    except ClusterDataIOError as e:
                        rlt = None
                        if str(e) == "skip":
                            pass
                        else:
                            io_error = True     
                    else:
                        if not is_read:
                            self.set_changed() # Marks the cluster as updated after writing operations.
                            if overwrited and log_type == self.LOG_ADD:
                                log_type = self.LOG_CHANGE
                            self.log_to_mark_file(log_type, src, dst, value)
                            if self.dataset_node is not None:
                                self.dataset_node.update_overview(log_type, src, dst, value, self)
                
                if io_error:
                    io_type_name = ["READ", "APPEND", "REMOVE", "CHANGE", "MOVE", "OPERATION"]
                    warning_str =   f"{self.__class__.__name__}:{self.mapping_name} " + \
                                    f"{io_type_name[log_type]}: src:{src}, dst:{dst}, value:{value} failed:" + \
                                    f"{warning_info}"
                    warnings.warn(warning_str, ClusterIONotExecutedWarning)

            return rlt
        return wrapper

    _FCT = TypeVar('_FCT', bound="FilesCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=FilesHandle)

    class _read(IOMeta[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value, **other_paras):
            if self.files_cluster.has(src):
                return self._query_fileshandle(src), None
            else:
                raise ClusterDataIOError

        def get_file_core_func(self, src_file_handle:FHT, dst_file_handle:FHT, value) -> Callable:
            return src_file_handle.read_func

        def io_cache(self, src_file_handle:FilesHandle, dst_file_handle, value=None) -> Any:
            if src_file_handle.has_cache:
                return src_file_handle.cache_proxy
            else:
                raise IOMetaPriorityError

        def cvt_to_core_paras(self, 
                              src_file_handle: FilesHandle, 
                              dst_file_handle: FilesHandle, 
                              value, 
                              **other_paras) -> tuple:
            return (src_file_handle.get_path(), )

    class _operation(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        LOG_TYPE = IOStatusManager.LOG_OPERATION

    class _write(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        PATH_EXISTS_REQUIRED = False
        LOG_TYPE = IOStatusManager.LOG_ADD

        def get_FilesHandle(self, src, dst, value,  **other_paras):
            if not self.files_cluster.has(dst):
                fh:FilesHandle = self.files_cluster.create_fileshandle(src, dst, value, **other_paras)
                self.files_cluster._set_fileshandle(dst, fh)
            return None, self._query_fileshandle(dst)

        def get_file_core_func(self, src_file_handle, dst_file_handle:FilesHandle, value) -> Callable[..., Any]:
            return dst_file_handle.write_func

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            dst_file_handle.set_cache(value)

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            dst_path = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return dst_path, value
        
    class _modify_key(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        LOG_TYPE = IOStatusManager.LOG_MOVE

        W_SYNC = True

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = os.rename

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            pass

        def get_FilesHandle(self, src, dst, value, **other_paras):
            src_handle:FilesHandle = self.files_cluster._pop_fileshandle(src) # type: ignore
            if not self.files_cluster.has(dst):

                dst_handle = self.files_cluster.FILESHANDLE_TYPE._ignore_create_same_path_fh_warning_once().\
                    from_fileshandle(self.files_cluster, src_handle, 
                        corename= self.files_cluster.format_corename(dst))
                self.files_cluster._set_fileshandle(dst, dst_handle)
            dst_handle = self._query_fileshandle(dst)
                
            return src_handle, dst_handle

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            src_path = src_file_handle.get_path()
            dst_path = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return src_path, dst_path

    class _remove(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        LOG_TYPE = IOStatusManager.LOG_REMOVE

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = os.remove

        def get_FilesHandle(self, src, dst, value, **other_paras):
            if not self.files_cluster.has(dst):
                fh = self.files_cluster.FILESHANDLE_TYPE.create_not_exist_fileshandle(self.files_cluster)
            else:
                fh = self.files_cluster._pop_fileshandle(dst)
            return None, fh

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            dst_file_handle.erase_cache()

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            return (dst_file_handle.get_path(), )

    class _paste_file(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        # W_SYNC = True
        PATH_EXISTS_REQUIRED = True
        LOG_TYPE = IOStatusManager.LOG_ADD

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = shutil.copy

        def get_FilesHandle(self, src, dst, value, **other_paras):
            src_handle:FilesHandle = value
            if not self.files_cluster.has(dst):
                dst_handle = self.files_cluster.FILESHANDLE_TYPE.from_fileshandle(self.files_cluster, src_handle, 
                                                                                  corename= self.files_cluster.format_corename(dst),
                                                                                  cache = FilesHandle.KW_INIT_WITHOUT_CACHE)
                self.files_cluster._set_fileshandle(dst, dst_handle)
            dst_handle = self._query_fileshandle(dst)
            return None, self._query_fileshandle(dst)

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value:FilesHandle=None) -> Any: # type: ignore
            dst_file_handle.cache = src_file_handle.cache

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value: FilesHandle, 
                                **other_paras) -> tuple:
            src_path = value.get_path()
            dst_path = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return src_path, dst_path

    class _change_dir(_operation[_FCT, _VDMT, _FHT]):
        W_SYNC = True
        
        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = os.rename

        def get_FilesHandle(self, src, dst, value:str):
            src_handle = self.files_cluster._pop_fileshandle(src)
            dst_handle = self.files_cluster.FILESHANDLE_TYPE.from_fileshandle(self.files_cluster, src_handle, sub_dir=value)
            self.files_cluster._set_fileshandle(src, dst_handle)
            return src_handle, dst_handle

        def io_cache(self, src_file_handle, dst_file_handle, value=None) -> Any:
            pass

        def cvt_to_core_paras(self,                                 
                              src_file_handle: FilesHandle, 
                              dst_file_handle: FilesHandle, 
                              value: FilesHandle,  
                              **other_paras) -> tuple:
            src_paths:Union[str, list[str]] = src_file_handle.get_path()
            dst_paths:Union[str, list[str]] = dst_file_handle.get_path()
            dst_dir = dst_file_handle.get_dir()
            os.makedirs(dst_dir, exist_ok=True)
            return src_paths, dst_paths
        
        def check_value(self, value: str):
            return isinstance(value, str)

    ### common operation ###
    @property
    def continuous(self):
        return self.num == self.i_upper

    @property
    def data_continuous(self):
        return self.data_num == self.data_i_upper
    
    @property
    def elem_continuous(self):
        return self.elem_num == self.elem_i_upper

    def has(self, i):
        return i in self.keys()

    def has_data(self, elem_i):
        return elem_i in self.data_keys()

    def has_elem(self, elem_i):
        return elem_i in self.elem_keys()
    
    @property
    def data_num(self):
        return len(self.data_keys())

    @property
    def data_i_upper(self):
        if self.KEY_TYPE != int:
            raise TypeError(f"the key_type of {self.__class__.__name__} is not int")
        return max(self.data_keys(), default = -1) + 1# type: ignore
    
    @property
    def elem_num(self):
        return len(self.elem_keys())
    
    @property
    def elem_i_upper(self):
        return max(self.elem_keys(), default=-1) + 1

    def __contains__(self, i):
        return self.has(i)
    
    def paste_file(self, dst:int, file_handler:FilesHandle, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.paste_file_meta, force)(dst = dst, value = file_handler, **other_paras)

    def change_dir(self, dst:int, new_dir_name, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.change_dir_meta, force)(dst = dst, value = new_dir_name, **other_paras)

    def cache_to_file(self, data_i:int = None, *, force = False, **other_paras):
        rlt = False
        self.write_meta._set_ctrl_flag(cache_priority=False, strict_priority_mode=True, 
                                       write_synchronous=False, io_raw=True)
        data_i_list = [data_i] if isinstance(data_i, int) else self.data_keys()
        for data_i in data_i_list:
            fh = self.MemoryData[data_i]
            if fh.synced or not fh.has_cache:
                continue
            value = fh.cache
            rlt = self.write_data(data_i, value, force = force, **other_paras)
            self.query_fileshandle(data_i).set_synced(True)
        self.write_meta._clear_ctrl_flag()
        return rlt

    def file_to_cache(self, data_i:int = None, *, save = True, force = False, **other_paras):
        rlt = False
        data_i_list = [data_i] if isinstance(data_i, int) else self.data_keys()

        self.read_meta._set_ctrl_flag(cache_priority=False, strict_priority_mode=True, 
                                        write_synchronous=False, io_raw=True)
        self.write_meta._set_ctrl_flag(cache_priority=True, strict_priority_mode=True, 
                                       write_synchronous=False, io_raw=True)

        for data_i in data_i_list:
            fh = self.query_fileshandle(data_i)
            if fh.synced or not fh.file_exist_status:
                continue
            value = self.read(data_i, force = force, **other_paras)
            if value is None:
                continue
            rlt = self.write_data(data_i, value, force = force, **other_paras)
            self.query_fileshandle(data_i).set_synced(True)

        self.write_meta._clear_ctrl_flag()
        self.read_meta._clear_ctrl_flag()

        if save:
            self.save()
        return rlt

    def clear_files(self, *, force = False):
        orig_cache_priority = self.cache_priority
        self.cache_priority = False
        self.clear(force = force, clear_both=False)
        self.cache_priority = orig_cache_priority

    def clear_cache(self, *, force = False):
        orig_cache_priority = self.cache_priority
        self.cache_priority = True
        self.clear(force = force, clear_both=False)
        self.cache_priority = orig_cache_priority
    ### common operation ###

    ### data operation ###

    def read_data(self, src:int, *, force = False, **other_paras) -> VDMT:
        return self.io_decorator(self.read_meta, force)(src = src, **other_paras) # type: ignore

    def write_data(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.write_meta, force)(dst = dst, value = value, **other_paras)
    
    def modify_data_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.modify_key_meta, force)(src = src, dst = dst, **other_paras)
    
    def remove_data(self, dst:int, remove_both = False, *, force = False, **other_paras) -> None:
        if remove_both:
            self.remove_meta._set_ctrl_flag(write_synchronous=True)
        rlt = self.io_decorator(self.remove_meta, force)(dst = dst, **other_paras)
        self.read_meta._clear_ctrl_flag()
        return rlt

    def merge_data_from(self, src_data_map:FCT, *, force = False):
        def merge_func(src_data_map:FCT, dst_data_map:FCT, data_i:int, data_i_upper:int):
            src_fh = src_data_map.query_fileshandle(data_i)
            dst_data_map.paste_file(
                data_i_upper,
                src_fh) # type: ignore
        
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        assert src_data_map.KEY_TYPE == int, f"the key_type of {src_data_map.__class__.__name__} is not int"
        assert type(src_data_map) == type(self), f"can't merge {type(src_data_map)} to {type(self)}"
        # assert self.opened and self.writable, f"{self.__class__.__name__} is not writable"
        assert src_data_map.opened, f"{src_data_map.__class__.__name__} is not opened"

        # if self.continuous:
        #     return

        with self.get_writer(force):
            for data_i in tqdm(src_data_map.keys(), desc=f"merge {src_data_map} to {self}", total=len(src_data_map)):
                merge_func(src_data_map, self, data_i, self.data_i_upper)

    def copy_data_from(self, src_data_map:FCT, *, cover = False, force = False):
        # if only_data_map:
        #     if os.path.exists(self.MemoryData_path):
        #         if cover:
        #             os.remove(self.MemoryData_path)
        #             shutil.copyfile(src_data_map.MemoryData_path, self.MemoryData_path)
        #             self.load()
        #         else:
        #             raise IOError(f"{self.MemoryData_path} already exists")
        # else:
        if os.path.exists(self.data_path) and len(self) > 0:
            if cover:
                shutil.rmtree(self.data_path)
                self.make_path()
                self.MemoryData.clear()
                self.save()
            else:
                raise IOError(f"{self.data_path} already exists")
        self.merge_from(src_data_map, force = force) 
    
    def clear_data(self, *, force = False, clear_both = True):
        if clear_both:
            with self.remove_meta._set_ctrl_flag(write_synchronous=True):
                super().clear(force = force)
        else:
            super().clear(force)
            for fh in self.query_fileshandle(0, 'end'):
                fh:FHT
                fh.set_synced(False)
    
    def data_keys(self):
        return self.MemoryData.keys()
    ### data operation ###
    
    ### elem operation ###

    def read_elem(self, src:int, *, force = False, **other_paras):
        raise NotImplementedError
        
    def write_elem(self, dst:int, value:VDMT, *, force = False, **other_paras):
        raise NotImplementedError

    def modify_elem_key(self, src:int, dst:int, *, force = False, **other_paras):
        raise NotImplementedError

    def remove_elem(self, dst:int, *, force = False, **other_paras):
        raise NotImplementedError
    
    def merge_elem_from(self, src_data_map:FCT, *, force = False):
        assert type(src_data_map) == type(self), f"src_data_map type {type(src_data_map)} != cluster type {type(self)}"
        assert len(src_data_map) == len(self), f"src_data_map length {len(src_data_map)} != cluster length {len(self)}"
        with self.get_writer(force).allow_overwriting():
            for elem_i in tqdm(src_data_map.elem_keys(), desc=f"merge {src_data_map} to {self}", total=src_data_map.elem_num):
                self.write_elem(self.elem_i_upper, src_data_map.read_elem(elem_i))
    
    def copy_elem_from(self, src_data_map:FCT, *, cover = False, force = False):
        if self.elem_num > 0:
            if cover:
                print("clearing elem...")
                self.clear_elem(force=True)
            else:
                raise DataMapExistError("can't copy to a non-empty cluster")
        self.merge_elem_from(src_data_map, force=force)
    
    def clear_elem(self, *, force = False, clear_both = True):
        with self.get_writer(force).allow_overwriting():
            for i in tqdm(list(self.elem_keys())):
                self.remove_elem(i)

    def elem_keys(self):
        raise NotImplementedError
    ### elem operation ###

    ### general ###    
    def _switch_io_operation(self, io_data_func, io_elem_func):
        if self._ELEM_BY_CACHE:
            return io_elem_func
        else:
            return io_data_func

    @property
    def num(self):
        if self._ELEM_BY_CACHE:
            return self.elem_num
        else:
            return self.data_num
    
    @property
    def i_upper(self):
        if self._ELEM_BY_CACHE:
            return self.elem_i_upper
        else:
            return self.data_i_upper

    def read(self, src:int, *, force = False, **other_paras) -> VDMT:
        return self._switch_io_operation(self.read_data, self.read_elem)(src = src, force = force, **other_paras)

    def write(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        return self._switch_io_operation(self.write_data, self.write_elem)(dst = dst, value = value, force = force, **other_paras)
        
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        return self._switch_io_operation(self.modify_data_key, self.modify_elem_key)(src = src, dst = dst, force = force, **other_paras)
        
    def remove(self, dst:int, remove_both = False, *, force = False, **other_paras) -> None:
        return self._switch_io_operation(self.remove_data, self.remove_elem)(dst = dst, remove_both = remove_both, force = force, **other_paras)

    def merge_from(self, src_data_map:FCT, *, force = False):
        return self._switch_io_operation(self.merge_data_from, self.merge_elem_from)(src_data_map = src_data_map, force = force)
    
    def copy_from(self, src_data_map:FCT, *, cover = False, force = False):
        return self._switch_io_operation(self.copy_data_from, self.copy_elem_from)(src_data_map = src_data_map, cover = cover, force = force)
    
    def clear(self, *, force = False, clear_both = True):
        rlt = self._switch_io_operation(self.clear_data, self.clear_elem)(force = force, clear_both = clear_both)
        if clear_both:
            self.remove_memory_data_file()
        return rlt
    
    def keys(self):
        return self._switch_io_operation(self.data_keys, self.elem_keys)()
    ###
    def idx_unwrited(self, idx):
        if self._ELEM_BY_CACHE:
            return not self.has_elem(idx)
        else:
            if self.has_data(idx):
                return self.query_fileshandle(idx).empty
            else:
                return True
    #### io metas END#####

    #### Memorydata operation ####
    def matching_path(self):
        paths:list[str] = []
        paths.extend(glob.glob(os.path.join(self.data_path, "**/*"), recursive=True))
        return paths

    def rebuild(self):
        paths:list[str] = self.matching_path()

        for path in paths:
            fh = self.FILESHANDLE_TYPE.from_path(self, path)
            data_i = self.deformat_corename(fh.corename)
            data_i = data_i if data_i is not None else self.data_i_upper
            if fh.all_file_exist:
                self._set_fileshandle(data_i, fh)
            else:
                self.paste_file(data_i, fh)

        for fh in list(self.MemoryData.values()):
            if fh.empty:
                self.remove(fh)

        self.sort()

    def merge_MemoryData(self, MemoryData:BinDict[int, FHT]):
        assert type(MemoryData) == self.MEMORY_DATA_TYPE, f"MemoryData type {type(MemoryData)} != cluster type {type(self)}"
        for loaded_fh in MemoryData.values():
            if loaded_fh in self.MemoryData._reverse_dict:
                this_fh = self.MemoryData[self.MemoryData._reverse_dict[loaded_fh]]
                assert loaded_fh.immutable_attr_same_as(this_fh), f"the fileshandle {loaded_fh} is not the same as {this_fh}"
                # cover cache
                this_fh.cache_proxy.cache = loaded_fh.cache_proxy.cache
            else:
                # add
                self._set_fileshandle(self.data_i_upper, loaded_fh)

    def save_preprecess(self, MemoryData:BinDict = None ):
        MemoryData = self.MemoryData if MemoryData is None else MemoryData
        to_save_dict = {item[0]: item[1].as_dict() for item in MemoryData.items()}
        return to_save_dict
    
    def load_postprocess(self, data:dict):
        new_dict = {int(k): self.FILESHANDLE_TYPE.from_dict(self, v) for k, v in data.items()}
        data_info_map = self.MEMORY_DATA_TYPE(new_dict)
        return data_info_map
    ###############

    @classmethod
    def from_cluster(cls:type[FCT], cluster:FCT, dataset_node:DSNT = None, name = None, *args, **kwargs) -> FCT:
        dataset_node    = cluster.dataset_node if dataset_node is None else dataset_node
        name            = cluster.mapping_name if name is None else name
        new_cluster:FCT = cls(dataset_node, name)
        return new_cluster

class DatasetNode(DataMapping[dict[str, bool], DSNT, VDST], ABC, Generic[FCT, DSNT, VDST]):
    MEMORY_DATA_TYPE = Table[int, str, bool]

    load_memory_func = JsonIO.load_json
    save_memory_func = JsonIO.dump_json 
    
    def init_identity(self, top_directory:str, name: str, *args, parent:"DatasetNode" = None, **kwargs):
        self.init_node_hook(parent)
        self._unfinished_operation = 0
        if self.parent is not None:
            self._top_directory = os.path.join(parent.data_path, top_directory)
        else:
            self._top_directory = top_directory
        self.mapping_name = name  

    def __init__(self, directory, *, parent:"DatasetNode" = None) -> None:
        if parent is not None:
            super().__init__(os.path.join(parent.data_path, directory), "")
        else:
            super().__init__(directory, "")
        
        self.init_dataset_attr()

        self.clusters_map:dict[str, FCT] = dict()
        self.init_clusters_hook()

        self.__inited = True # if the dataset has been inited

    @property
    def MemoryData(self) -> Table[int, str, bool]:
        return self._MemoryData

    def init_node_hook(self, parent):
        self.parent:DatasetNode = parent
        self.children:list[DatasetNode] = []
        self.move_node(parent)

        self.linked_with_children = True
        self.follow_parent = True

    def init_dataset_attr(self):
        def is_directory_inside(base_dir, target_dir):
            base_dir:str = os.path.abspath(base_dir)
            target_dir:str = os.path.abspath(target_dir)
            return target_dir.startswith(base_dir)
        
        os.makedirs(self.top_directory, exist_ok=True) 
        self._unfinished_operation = 0
        if self.parent is not None:
            assert is_directory_inside(self.parent.top_directory, self.top_directory), f"{self.top_directory} is not inside {self.parent.top_directory}"
            self.name:str = os.path.relpath(self.top_directory, self.parent.top_directory)
        else:
            self.name:str = self.top_directory

    #### cluster_map     ####
    def add_cluster(self, cluster:FCT):
        cluster._dataset_node = self
        if cluster.mapping_name in self.clusters_map:
            raise KeyError(f"{cluster.mapping_name} already exists")
        else:
            self.clusters_map[cluster.mapping_name] = cluster
        if self.opened:
            cluster.open()
            self.MemoryData.col_names
        if cluster._IS_ELEM:
            self.MemoryData.add_column(cluster.mapping_name, exist_ok=True)

    def remove_cluster(self, cluster:FCT):
        if cluster.dataset_node == self:
            cluster.dataset_node = None
            del self.clusters_map[cluster.mapping_name]
            self.MemoryData.remove_column(cluster.mapping_name, not_exist_ok=True)

    def get_cluster(self, name:str):
        return self.clusters_map[name]

    def cluster_keys(self):
        return self.clusters_map.keys()

    @property
    def children_names(self):
        return [x.name for x in self.children]

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        raise KeyError(f"{name} not in {self}")

    @property
    def clusters(self) -> list[FCT]:
        clusters = list(self.clusters_map.values())
        return clusters

    @property
    def elem_clusters(self) -> list[FCT]:
        clusters = [x for x in self.clusters if x._IS_ELEM]
        return clusters

    @property
    def opened_clusters(self):
        clusters = [x for x in self.clusters if x.opened]
        return clusters

    @property
    def opened_elem_clusters(self):
        clusters = [x for x in self.elem_clusters if x.opened]
        return clusters
    
    @property
    def clusters_num(self) -> int:
        return len(self.clusters_map)

    def get_all_clusters(self, _type:Union[type, tuple[type]] = None, only_opened = False) -> dict[int, FCT]:
        cluster_map:dict[str, FCT] = {}

        for k in self.cluster_keys():
            if k in cluster_map:
                raise NotImplementedError(f"the mapping name {k} is already in cluster_map")
            cluster_map[k] = self.clusters_map[k]

        if self.linked_with_children:
            for k, v in copy.copy(cluster_map.items()):
                if _type is not None:
                    if not isinstance(v, _type):
                        cluster_map.pop(k)
                if only_opened:
                    if v.opened:
                        cluster_map.pop(k)

        for child in self.children:
            cluster_map.update(child.get_all_clusters(_type, only_opened))

        return cluster_map
    #### cluster_map END ####
    def operate_clusters(self, func:Union[Callable, str], *args, **kwargs):
        for obj in self.clusters:
            self.operate_one_cluster(obj, func, *args, **kwargs)

    def operate_children_node(self, func:Union[Callable, str], *args, **kwargs):
        for child_node in self.children:
            self.operate_one_child_node(child_node, func, *args, **kwargs)

    def operate_one_cluster(self, cluster, func:Union[Callable, str], *args, **kwargs):
        func_name = func.__name__ if isinstance(func, Callable) else func
        cluster.__getattribute__(func_name)(*args, **kwargs)

    def operate_one_child_node(self, child_node:"DatasetNode", func:Union[Callable, str], *args, **kwargs):
        func_name = func.__name__ if isinstance(func, Callable) else func
        if self.linked_with_children and child_node.follow_parent:
            child_node.__getattribute__(func_name)(*args, **kwargs)

    def open_hook(self):
        super().open_hook()
        self.operate_clusters(FilesCluster.open)
        self.operate_children_node(DatasetNode.open)

    def close_hook(self):
        super().close_hook()
        self.operate_clusters(FilesCluster.close)
        self.operate_children_node(DatasetNode.close)

    def readonly_hook(self):
        super().readonly_hook()
        self.operate_clusters(FilesCluster.set_readonly)
        self.operate_children_node(DatasetNode.set_readonly)

    def writable_hook(self):
        super().writable_hook()
        self.operate_clusters(FilesCluster.set_writable)
        self.operate_children_node(DatasetNode.set_writable)

    def stop_writing_hook(self):
        super().stop_writing_hook()
        self.operate_clusters(FilesCluster.stop_writing)
        self.operate_children_node(DatasetNode.stop_writing)

    def start_writing_hook(self):
        super().start_writing_hook()
        self.operate_clusters(FilesCluster.start_writing)
        self.operate_children_node(DatasetNode.start_writing)

    def set_overwrite_allowed_hook(self):
        super().set_overwrite_allowed_hook()
        self.operate_clusters(FilesCluster.set_overwrite_allowed)
        self.operate_children_node(DatasetNode.set_overwrite_allowed)

    def set_overwrite_forbidden_hook(self):
        super().set_overwrite_forbidden_hook()
        self.operate_clusters(FilesCluster.set_overwrite_forbidden)
        self.operate_children_node(DatasetNode.set_overwrite_forbidden)
    ####################

    ##### clusters #####
    def init_clusters_hook(self):
        pass
        # unfinished = self.mark_exist()
        # if unfinished:
        #     y:int = FilesCluster.choose_unfinished_operation(self)
        #     self._unfinished_operation = y        

        # self._init_clusters()
        # self.update_dataset()        

        # self.load_overview()        
        # if unfinished:
        #     self.process_unfinished()
        #     os.remove(self.get_writing_mark_file())

    def __setattr__(self, name, value):
        ### 同名变量赋值时，自动将原有对象解除注册
        if name in self.__dict__:
            obj = self.__getattribute__(name)
            if isinstance(obj, DatasetNode):
                if value is not None:
                    assert isinstance(value, DatasetNode), f"the type of {name} must be DatasetNode, not {type(value)}"
                if obj.parent == self:
                    obj.move_node(None)
            elif isinstance(obj, FilesCluster):
                assert isinstance(value, FilesCluster), f"the type of {name} must be FilesCluster, not {type(value)}"
                if obj.dataset_node == self:
                    obj.unregister_from_dataset()
        super().__setattr__(name, value)
    ####################

    ### node ###
    def add_child(self, child_node:"DatasetNode"):
        assert isinstance(child_node, DatasetNode), f"child_node must be Node, not {type(child_node)}"
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node:"DatasetNode"):
        assert isinstance(child_node, DatasetNode), f"child_node must be Node, not {type(child_node)}"
        if child_node in self.children:
            child_node.parent = None
            self.children.remove(child_node)

    def move_node(self, new_parent:"DatasetNode"):
        assert isinstance(new_parent, DatasetNode) or new_parent is None, f"new_parent must be DatasetNode, not {type(new_parent)}"
        if self.parent is not None:
            self.parent.remove_child(self)
        if new_parent is not None:
            new_parent.add_child(self)
    ############

    ############
    def raw_read(self, src, *, force = False, **other_paras) -> VDST:
        read_rlt = []
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                rlt = obj.read(src, **other_paras)
                read_rlt.append(rlt)
        return read_rlt
    
    def raw_write(self, dst, values:list, *, force = False, **other_paras) -> None:
        assert len(values) == len(self.elem_clusters)
        with self.get_writer(force).allow_overwriting():
            for i, obj in enumerate(self.elem_clusters):
                obj.write(dst, values[i], **other_paras)

    def read(self, src:int):
        return self.raw_read(src)

    def write(self, dst:int, value:VDST, *, force = False, **other_paras) -> None:
        return self.raw_write(dst, value, force = force, **other_paras)
        
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                obj.modify_key(src, dst, **other_paras)

    def remove(self, dst:int, *, force = False, **other_paras) -> None:
        with self.get_writer(force).allow_overwriting():
            for obj in self.elem_clusters:
                obj.remove(dst, **other_paras)

    def merge_from(self, src_dataset_node:DSNT, *, force = False) -> None:
        assert type(src_dataset_node) == type(self), f"can't merge {type(src_dataset_node)} to {type(self)}"
        if self.clusters_num > 0:
            assert src_dataset_node.clusters_num == self.clusters_num, f"the clusters_num of {src_dataset_node} is not equal to {self}"
       
        if self.clusters_num == 0:
            # add clusters
            for cluster_name in src_dataset_node.cluster_keys():
                src_cluster = src_dataset_node.clusters_map[cluster_name]
                self.add_cluster(src_cluster.__class__.from_cluster(src_cluster, dataset_node=self))

        with self.get_writer(force).allow_overwriting():  
            for cluster_name in self.cluster_keys():
                this_cluster = self.clusters_map[cluster_name]
                src_cluster = src_dataset_node.clusters_map[cluster_name]
                self.operate_one_cluster(this_cluster, FilesCluster.merge_from, src_cluster)
            for this_child_node in self.children:
                src_child_node = src_dataset_node.get_child(this_child_node.name)
                self.operate_one_child_node(this_child_node, DatasetNode.merge_from, src_child_node)

    def copy_from(self, src_dataset_node:DSNT, *, cover = False, force = False) -> None:
        if self.num > 0:
            if cover:
                print(f"clear {self}")
                self.clear(force = force)
            else:
                raise DataMapExistError(f"{self} is not empty")
        self.merge_from(src_dataset_node, force = force)

    def clear(self, *, force = False, clear_both = True, clear_completely = False):
        if clear_completely:
            self.close()
            shutil.rmtree(self.data_path)
        else:
            with self.get_writer(force).allow_overwriting():
                self.operate_clusters(DataMapping.clear, clear_both = clear_both)
                self.operate_children_node(DataMapping.clear, clear_both = clear_both)
    ############

    ###### cache operation ######
    def all_cache_to_file(self, *, force = False):
        self.operate_clusters(FilesCluster.cache_to_file, force=force)
        self.operate_children_node(DatasetNode.all_cache_to_file, force=force)

    def all_file_to_cache(self, *, force = False):
        self.operate_clusters(FilesCluster.file_to_cache, force=force)
        self.operate_children_node(DatasetNode.all_file_to_cache, force=force)
    ############

    ############
    def update_overview(self, log_type, src, dst, value, cluster:FilesCluster):
        col_name = cluster.mapping_name
        if log_type == self.LOG_READ or\
        log_type == self.LOG_CHANGE or\
        log_type == self.LOG_OPERATION:
            return
        if log_type == self.LOG_ADD:
            self.MemoryData.add_row(dst, exist_ok=True)
            self.MemoryData[dst, col_name] = True
        if log_type == self.LOG_REMOVE:
            self.MemoryData[dst, col_name] = False
            self.clear_empty_row(dst)
        if log_type == self.LOG_MOVE:
            self.MemoryData.add_row(dst, exist_ok=True)
            self.MemoryData[dst, col_name] = True
            self.MemoryData[src, col_name] = False
            self.clear_empty_row(src)

    def rebuild(self):
        if len(self.elem_clusters) > 0:
            rows = [x for x in range(self.i_upper)]
            cols = [x.mapping_name for x in self.elem_clusters]
            self._MemoryData = Table(rows, cols, bool, row_name_type=int, col_name_type=str)
            for data_i in tqdm(self.MemoryData.data, desc="initializing data frame"):
                self.calc_overview(data_i)

    def merge_MemoryData(self, MemoryData:Table[int, str, bool]):
        self.MemoryData.merge(MemoryData)

    def save_preprecess(self, MemoryData:Table = None ):
        MemoryData = self.MemoryData if MemoryData is None else MemoryData
        to_save_dict = dict(MemoryData.data)
        for row_i in MemoryData.data.keys():
            if not any(to_save_dict[row_i].values()):
                to_save_dict.pop(row_i)
        return to_save_dict
    
    def load_postprocess(self, data:dict):
        data_info_map = self.MEMORY_DATA_TYPE(default_value_type=bool, row_name_type=int, col_name_type=str, data=data)
        return data_info_map

    def calc_overview(self, data_i):
        self.MemoryData.add_row(data_i, exist_ok=True)        
        for cluster in self.elem_clusters:
            self.MemoryData[data_i, cluster.mapping_name] = cluster.has(data_i)

    def clear_empty_row(self, data_i):
        if not any(self.MemoryData[data_i].values()):
            self.MemoryData.remove_row(data_i)
    ############

def parse_kw(**kwargs) -> list[dict[str, Any]]:
    if len(kwargs) == 0:
        return []
    assert all([isinstance(v, list) for v in kwargs.values()])
    length = len(list(kwargs.values())[0])
    assert all([len(v) == length for v in kwargs.values()])
    kw_keys = list(kwargs.keys())
    kw_values = list(kwargs.values())
    
    kws = []
    for data_i in range(length):
        kws.append({k:v[data_i] for k, v in zip(kw_keys, kw_values)})

    return kws