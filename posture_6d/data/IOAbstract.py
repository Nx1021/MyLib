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
import copy

from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt,\
      serialize_object, deserialize_object, read_file_as_str, write_str_to_file
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

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

class IOStatusWarning(Warning):
    pass

class ClusterIONotExecutedWarning(ClusterWarning):
    pass

class ClusterNotRecommendWarning(ClusterWarning):
    pass

class IOStatusManager():
    WRITING_MARK = '.writing'

    LOG_READ = 0
    LOG_APPEND = 1
    LOG_REMOVE = 2
    LOG_CHANGE = 3
    LOG_MOVE   = 4
    LOG_OPERATION = 5
    LOG_KN = [LOG_READ, LOG_APPEND, LOG_REMOVE, LOG_CHANGE, LOG_MOVE, LOG_OPERATION]

    def __init__(self, name) -> None:
        self.name = name

        self.__closed = True
        self.__readonly = True
        self.__wait_writing = True
        self.__overwrite_allowed = False

        self.__writer = self._IOContext(self, True, True)

    class _IOContext():
        def __init__(self, 
                     obj:"IOStatusManager", 
                     open = False, 
                     writing = False, 
                     overwrite_allowed = False) -> None:
            self.obj:IOStatusManager = obj
            self.orig_readonly = True
            self.orig_overwrite_allowed = True
            self.open = open
            self.writing = writing
            self.overwrite_allowed = overwrite_allowed

            self.__valid = True

        def allow_overwriting(self):
            self.overwrite_allowed = True
            return self

        def valid(self, valid):
            self.__valid = valid
            return self

        def __enter__(self):
            if self.__valid:
                self.orig_readonly              = self.obj.readonly
                self.orig_overwrite_allowed     = self.obj.overwrite_allowed
                self.obj.open()
                self.obj.set_writable(self.writing)
                if self.writing:
                    self.obj.start_writing()
                self.obj.set_overwrite_allowed(self.overwrite_allowed)
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                self.__valid = True
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                if self.__valid:
                    self.obj.set_overwrite_allowed(self.orig_overwrite_allowed)
                    if self.obj.is_writing:
                        self.obj.stop_writing()
                    self.obj.set_readonly(self.orig_readonly)
                self.overwrite_allowed = False
                self.__valid = True
                return True

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

    @property
    def writer(self):
        return self.__writer

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
            self.stop_writing()
            self.set_readonly()
            self.close_hook()
            self.__closed = closed
        elif self.__closed and not closed:
            self.__closed = closed
            self.open_hook()

    def open(self, opened:bool = True):
        self.close(not opened)

    def set_readonly(self, readonly:bool = True):
        if self.closed:
            warnings.warn(f"the Status is closed, please call '{self.set_readonly.__name__}' when it's opened", IOStatusWarning)
        if not self.__readonly and readonly:
            self.stop_writing()
            self.readonly_hook()
        elif self.__readonly and not readonly:
            self.writable_hook()
        self.__readonly = readonly
        
    def set_writable(self, writable:bool = True):
        self.set_readonly(not writable)    

    def stop_writing(self, stop_writing:bool = True):
        if self.closed or self.readonly:
            warnings.warn(f"the Status is closed or readonly, please call '{self.stop_writing.__name__}' when it's writable", IOStatusWarning)
        if not self.__wait_writing and stop_writing:
            self.set_overwrite_forbidden()
            self.stop_writing_hook()
            if os.path.exists(self.get_writing_mark_file()):
                os.remove(self.get_writing_mark_file())
            self.__wait_writing = True
        elif self.__wait_writing and not stop_writing:
            self.__wait_writing = False
            with open(self.get_writing_mark_file(), 'w'):
                pass
            self.start_writing_hook()
        self.__wait_writing = stop_writing

    def start_writing(self, start_writing:bool = True, overwrite_allowed:bool = False):
        self.set_overwrite_allowed(overwrite_allowed)
        self.stop_writing(not start_writing)

    def set_overwrite_allowed(self, overwrite_allowed:bool = True):
        if self.closed or self.readonly:
            warnings.warn(f"the Status is closed or readonly, please call '{self.set_overwrite_allowed.__name__}' when it's writable", IOStatusWarning)
        if not self.__overwrite_allowed and overwrite_allowed:
            self.set_overwrite_allowed_hook()
        elif self.__overwrite_allowed and not overwrite_allowed:
            self.set_overwrite_forbidden_hook()
        self.__overwrite_allowed = overwrite_allowed

    def set_overwrite_forbidden(self, overwrite_forbidden:bool = True):
        self.set_overwrite_allowed(not overwrite_forbidden)
    
    def is_closed(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self.__closed:
            warnings.warn(f"{self.__class__.__name__}:{self.name} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self.__closed

    def is_readonly(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self.__readonly:
            warnings.warn(f"{self.__class__.__name__}:{self.name} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self.__readonly
    
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

# class FilesHandle(dict[str, Union[dict, str, Callable, type, None]], Generic[FCT]):
#     '''
#     immutable object, once created, it can't be changed.
#     '''
#     KW_top_directory     = "top_directory"
#     KW_sub_dir           = "sub_dir"
#     KW_corename          = "corename"
#     KW_suffix            = "suffix"
#     KW_appendnames       = "appendnames"   
#     KW_prefix            = "prefix" 
#     KW_read_func         = "read_func"
#     KW_write_func        = "write_func"
#     KW_value_type        = "value_type"

#     KW_cache             = "cache"  

#     KW_SLAVE = {
#         KW_top_directory:    True,
#         KW_sub_dir      :    False,
#         KW_corename     :    False,
#         KW_suffix       :    False,        
#         KW_read_func    :    False,
#         KW_write_func   :    False,
#         KW_value_type   :    False
#     }

#     @classmethod
#     @property
#     def top_directory_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_top_directory]
    
#     @classmethod
#     @property
#     def sub_dir_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_sub_dir]
    
#     @classmethod
#     @property
#     def corename_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_corename]
    
#     @classmethod
#     @property
#     def suffix_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_suffix]

#     @classmethod
#     @property
#     def read_func_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_read_func]
    
#     @classmethod
#     @property
#     def write_func_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_write_func]
    
#     @classmethod
#     @property
#     def value_type_as_slave(cls):
#         return cls.KW_SLAVE[cls.KW_value_type]

#     def __init__(self, cluster:FCT, sub_dir:str, corename:str, suffix:str, * ,
#                  prefix:str = "", appendnames:list[str] = None,  # type: ignore
#                  prefix_joiner:str = '', appendnames_joiner:str = '',
#                  top_directory = "",
#                  read_func:Callable = None, write_func:Callable = None, value_type:Callable = None) -> None: #type: ignore
#         super().__init__()
#         appendnames = appendnames if appendnames is not None else ['']
#         self.cluster:FCT = cluster
#         self[self.KW_top_directory] = top_directory
#         self[self.KW_sub_dir] = sub_dir
#         self[self.KW_corename] = corename
#         self[self.KW_suffix] = suffix

#         self[self.KW_prefix] = _Prefix(prefix, prefix_joiner)
#         self[self.KW_appendnames] = _AppendNames(appendnames, appendnames_joiner)

#         self[self.KW_read_func] = read_func
#         self[self.KW_write_func] = write_func
#         self[self.KW_value_type] = value_type

#         self[self.KW_cache] = None
#         self.synced = False
#         self.cache_io_mode = 'r'

#     def __hash__(self):
#         return id(self)
    
#     def __eq__(self, o: object) -> bool:
#         if isinstance(o, FilesHandle):
#             return hash(self) == hash(o)
#         else:
#             return False

#     def __enter__(self):
#         self.cache_io_mode = 'w'
#         self.synced = False
#         return self
    
#     def __exit__(self, exc_type, exc_value, traceback):
#         if exc_type is not None:
#             raise exc_type(exc_value).with_traceback(traceback)
#         else:
#             self.cache_io_mode = 'r'

#     def __getattribute__(self, name):
#         value = super().__getattribute__(name)
#         kw_slave:dict = super().__getattribute__('KW_SLAVE')
#         if name in kw_slave.keys() and kw_slave[name]:
#             self[name] = value
#         return value

#     def __setattr__(self, name, value):
#         if name in self.KW_SLAVE.keys() and self.KW_SLAVE[name]:
#             return None
#         else:
#             super().__setattr__(name, value)

#     @property
#     def multi_files(self):
#         return self.cluster.MULTI_FILES

#     @property
#     def top_directory(self) -> str:
#         if self.top_directory_as_slave:
#             return os.path.join(self.cluster.top_directory, self.cluster.mapping_name)
#         else:
#             return self[self.KW_top_directory] # type: ignore
    
#     @top_directory.setter
#     def top_directory(self, value):
#         assert isinstance(value, str), f"top_directory must be a str"
#         self[self.KW_top_directory] = value

#     @property
#     def sub_dir(self) -> str:
#         return self[self.KW_sub_dir] # type: ignore
    
#     @sub_dir.setter
#     def sub_dir(self, value):
#         assert isinstance(value, str), f"sub_dir must be a str"
#         self[self.KW_sub_dir] = value

#     @property
#     def corename(self) -> str:
#         return self[self.KW_corename] # type: ignore
    
#     @corename.setter
#     def corename(self, value):
#         assert isinstance(value, str), f"corename must be a str"
#         self[self.KW_corename] = value

#     @property
#     def _prefix_obj(self) -> _Prefix:
#         return self[self.KW_prefix] # type: ignore

#     @property
#     def prefix(self) -> str:
#         return self._prefix_obj.prefix

#     @property
#     def prefix_with_joiner(self) -> str:
#         return self._prefix_obj.get_with_joiner()

#     @prefix.setter
#     def prefix(self, value):
#         self._prefix_obj.prefix = value    

#     @property
#     def _appendnames_obj(self) -> _AppendNames:
#         return self[self.KW_appendnames] # type: ignore

#     @property
#     def appendnames(self) -> Union[list[str], str]:
#         appendnames = self._appendnames_obj.appendnames
#         return _AppendNames.conditional_return(self.multi_files, appendnames)

#     @property
#     def appendnames_with_joiner(self) -> Union[list[str], str]:
#         apwj = self._appendnames_obj.get_with_joiner()
#         return _AppendNames.conditional_return(self.multi_files, apwj)

#     @appendnames.setter
#     def appendnames(self, value:Union[list[str], str]):
#         assert isinstance(value, (list, str)), f"appendnames must be a list or str"
#         self._appendnames_obj.extend(value)

#     @property
#     def suffix(self) -> str:
#         return self[self.KW_suffix] # type: ignore
    
#     @suffix.setter
#     def suffix(self, value):
#         assert isinstance(value, str), f"suffix must be str, not {type(value)}"
#         if value[0] != '.':
#             value = '.' + value
#         self[self.KW_suffix] = value

#     @property
#     def read_func(self) -> Callable:
#         return self[self.KW_read_func] # type: ignore
    
#     @read_func.setter
#     def read_func(self, value):
#         assert isinstance(value, Callable), f"read_func must be a Callable"
#         self[self.KW_read_func] = value

#     @property
#     def write_func(self) -> Callable:
#         return self[self.KW_write_func] # type: ignore
    
#     @write_func.setter
#     def write_func(self, value):
#         assert isinstance(value, Callable), f"write_func must be a Callable"
#         self[self.KW_write_func] = value

#     @property
#     def value_type(self) -> type:
#         return self[self.KW_value_type] # type: ignore
    
#     @value_type.setter
#     def value_type(self, value_type):
#         assert isinstance(value_type, type), f"value_type must be a type"
#         if value_type is not None:
#             if self.cache is not None:
#                 assert type(self.cache) == value_type, f"the type of cache must be {value_type}, not {type(self.cache)}"
#             else:
#                 self[self.KW_value_type] = value_type
#         else:
#             self[self.KW_value_type] = None

#     @property
#     def cache(self) -> Any:
#         if self.cache_io_mode == 'r':
#             return copy.copy(self[self.KW_cache])
#         else:
#             return self[self.KW_cache]
    
#     @cache.setter
#     def cache(self, value):
#         if self.cache_io_mode == 'r':
#             warnings.warn(f"cache is read-only, please use 'with' statement to write cache", IOStatusWarning)
#         else:
#             if self.value_type is not None:
#                 assert isinstance(value, self.value_type), f"the type of cache must be {self.value_type}, not {type(value)}"
#             self[self.KW_cache] = value
#             self.synced = False

#     @property
#     def full_directory(self):
#         return os.path.join(self.top_directory, self.sub_dir)

#     @property
#     def all_file_exist(self):
#         paths = self.get_path(get_list = True)
#         if len(paths) == 0:
#             return False
#         return all([os.path.exists(x) for x in paths])
    
#     @property
#     def all_file_not_exist(self):
#         paths = self.get_path(get_list = True)
#         if len(paths) == 0:
#             return True
#         return all([not os.path.exists(x) for x in paths])
    
#     @property
#     def any_file_exist(self):
#         paths = self.get_path(get_list = True)
#         if len(paths) == 0:
#             return False
#         return any([os.path.exists(x) for x in paths])
    
#     @property
#     def any_file_not_exist(self):
#         paths = self.get_path(get_list = True)
#         if len(paths) == 0:
#             return True
#         return any([not os.path.exists(x) for x in paths])
    
#     @property
#     def has_cache(self):
#         return self.cache is not None

#     @property
#     def empty(self):
#         return not (self.any_file_exist or self.has_cache)

#     @property
#     def file_exist_status(self) -> list[bool]:
#         paths = self.get_path(get_list=True)
#         return [os.path.exists(x) for x in paths]

#     @property
#     def valid(self):
#         return self.cluster is not None

#     def set_corename_by_format(self, data_i:int):
#         corename = self.cluster.format_corename(data_i)
#         if corename is not None:
#             self.corename = corename
        
#     def get_name(self, get_list = False) -> Union[list[str], str]:
#         if self.multi_files or get_list:
#             awu_list:list[str] = self._appendnames_obj.get_with_joiner() 
#             return [self.prefix_with_joiner + self.corename + x + self.suffix for x in awu_list]
#         else:
#             awu:str = self.appendnames_with_joiner # type: ignore
#             return self.prefix_with_joiner + self.corename + awu + self.suffix 

#     def get_path(self, get_list = False) -> Union[list[str], str]:
#         name = self.get_name(get_list = get_list)
#         if isinstance(name, str):
#             return os.path.join(self.top_directory, self.sub_dir, name)
#         else:
#             return [os.path.join(self.top_directory, self.sub_dir, x) for x in name]
    
#     def as_dict(self):
#         for kw, as_slave in self.KW_SLAVE.items():
#             if as_slave:
#                 self.__getattribute__(kw)

#         dict_ = dict(self)
#         dict_[self.KW_prefix]       = self._prefix_obj.as_dict()
#         dict_[self.KW_appendnames]  = self._appendnames_obj.as_dict()

#         return dict_

#     @classmethod
#     def from_dict(cls, cluster, dict_:dict):
#         top_directory   = dict_[cls.KW_top_directory]
#         sub_dir         = dict_[cls.KW_sub_dir]
#         corename        = dict_[cls.KW_corename]
#         suffix          = dict_[cls.KW_suffix]      

#         prefix                  = dict_[cls.KW_prefix][_Prefix.KW_PREFIX]
#         prefix_joiner           = dict_[cls.KW_prefix][_Prefix.KW_JOINER]
#         appendnames             = dict_[cls.KW_appendnames][_AppendNames.KW_APPENDNAMES]
#         appendnames_joiner      = dict_[cls.KW_appendnames][_AppendNames.KW_JOINER]

#         cache           = dict_[cls.KW_cache]
#         read_func       = dict_[cls.KW_read_func]
#         write_func      = dict_[cls.KW_write_func]
#         value_type      = dict_[cls.KW_value_type]
#         obj = cls(cluster, sub_dir, corename, suffix,
#             prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
#             top_directory = top_directory,
#             read_func = read_func, write_func = write_func, value_type = value_type)
#         with obj:
#             obj.cache = cache
#         return obj
    
#     @classmethod
#     def from_path(cls, cluster:FCT, path:Union[str, list[str]], *,
#                   prefix_joiner:str = '', appendnames_joiner:str = '', 
#                   read_func:Callable = None, write_func:Callable = None, value_type:Callable = None,  #type: ignore
#                   _extract_corename_func:Callable[[str], tuple[str, str, str, str, str]] = None): #type: ignore
#         def parse_one(path:str):
#             top_directory = cluster.top_directory
#             sub_dir, name = os.path.relpath(path, top_directory).split(os.sep)
#             basename, suffix = os.path.splitext(name)

#             if _extract_corename_func is not None:
#                 corename, prefix, appendname, _prefix_joiner, _appendnames_joiner = _extract_corename_func(basename)
#             else:
#                 _prefix_joiner = prefix_joiner
#                 _appendnames_joiner = appendnames_joiner
#                 if prefix_joiner == "":
#                     prefix, rest = "", basename
#                 else:
#                     prefix, rest = basename.split(prefix_joiner, 1)

#                 if appendnames_joiner == "":
#                     corename, appendname = rest, ""
#                 else:
#                     corename, appendname = rest.split(appendnames_joiner, 1)

#             return top_directory, sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner
        
#         if isinstance(path, str):
#             top_directory, sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner = parse_one(path)
#             appendnames = list[str]([appendname])
#         else:
#             assert len(path) > 0, f"path must be a str or a list of str"
#             top_directory = ""
#             _, sub_dirs, corenames, suffixes, prefixes, appendnames = [], [], [], [], [], []
#             for p in path:
#                 top_directory, sub_dir, corename, suffix, prefix, appendname, _prefix_joiner, _appendnames_joiner = parse_one(p)
#                 sub_dirs.append(sub_dir)
#                 corenames.append(corename)
#                 suffixes.append(suffix)       
#                 prefixes.append(prefix)         
#                 appendnames.append(appendname)

#             assert len(set(sub_dirs)) == 1, f"sub_dir must be the same"
#             assert len(set(corenames)) == 1, f"corename must be the same"
#             assert len(set(suffixes)) == 1, f"suffix must be the same"
#             assert len(set(prefixes)) == 1, f"prefix must be the same"
#             sub_dir = sub_dirs[0]
#             corename = corenames[0]
#             suffix = suffixes[0]
#             prefix = prefixes[0]
#         return cls(cluster, sub_dir, corename, suffix,
#                    prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
#                    top_directory = top_directory,
#                    read_func = read_func, write_func = write_func, value_type = value_type)
    
#     @classmethod
#     def from_fileshandle(cls, cluster, file_handle:"FilesHandle", *,
#                         sub_dir:str = None, corename:str = None, suffix:str = None, #type: ignore
#                         prefix:str = None, appendnames:list[str] = None, prefix_joiner:str = None, appendnames_joiner:str = None, #type: ignore
#                         read_func:Callable = None, write_func:Callable = None, value_type:Callable = None): #type: ignore
#         sub_dir = file_handle.sub_dir if sub_dir is None else sub_dir
#         corename = file_handle.corename if corename is None else corename
#         suffix = file_handle.suffix if suffix is None else suffix

#         prefix = file_handle.prefix if prefix is None else prefix
#         appendnames = file_handle._appendnames_obj.appendnames if appendnames is None else appendnames
#         prefix_joiner = file_handle._prefix_obj.joiner if prefix_joiner is None else prefix_joiner
#         appendnames_joiner = file_handle._appendnames_obj.joiner if appendnames_joiner is None else appendnames_joiner

#         read_func = file_handle.read_func if read_func is None else read_func
#         write_func = file_handle.write_func if write_func is None else write_func
#         value_type = file_handle.value_type if value_type is None else value_type
        
#         return cls(cluster, sub_dir, corename, suffix,
#                      prefix = prefix, appendnames = appendnames, prefix_joiner = prefix_joiner, appendnames_joiner = appendnames_joiner,
#                         top_directory = file_handle.top_directory,
#                     read_func = read_func, write_func = write_func, value_type = value_type)
    
#     def add_appendname(self, appendname:str):
#         if appendname not in self.appendnames:
#             self._appendnames_obj.appendnames.append(appendname)

#     def remove_appendname(self, appendname:str):
#         if appendname in self.appendnames:
#             self._appendnames_obj.appendnames.remove(appendname)

#     def set_appendname(self, appendname:str):
#         assert not self.multi_files, f"set_appendname is not allowed in multi_files mode"
#         self._appendnames_obj.appendnames.clear()
#         self._appendnames_obj.appendnames.append(appendname)

#     def clear_notfound(self):
#         new_appendnames = []
#         for i, e in enumerate(self.file_exist_status):
#             if e:
#                 new_appendnames.append(self._appendnames_obj.appendnames[i])
#         self.appendnames = new_appendnames

#     def sort(self):
#         new_appendnames = []
#         for i in sorted(self.appendnames):
#             idx = self.appendnames.index(i)
#             new_appendnames.append(i)
#         self.appendnames = new_appendnames

#     def erase_cache(self):
#         if self.cache_io_mode == 'r':
#             warnings.warn(f"cache is read-only, please use 'with' statement to write cache", IOStatusWarning)
#         else:
#             self.cache = None
#             self.synced = False

#     def get_key(self):
#         return self.cluster._MemoryData._reverse_dict[self] # type: ignore

#     def __repr__(self) -> str:
#         if len(self._appendnames_obj.appendnames) == 1:
#             string = f"FilesHandle({self.get_path()})"
#         else:
#             paths = self.get_path(get_list=True)
#             string = f"FilesHandle({paths[0]}) + {len(paths) - 1} files"
#         string += f" file exist: {self.file_exist_status}, has cache: {self.has_cache}, synced: {self.synced}"
#         return string

class CacheProxy():
    KW_cache = "cache"
    KW_value_type = "value_type"

    def __init__(self, cache, value_type = None) -> None:
        self.__cache = cache
        self.synced = False
        self.__value_type = value_type
        
    @property
    def value_type(self) -> type:
        return self.__value_type # type: ignore
    
    @value_type.setter
    def value_type(self, value_type):
        assert isinstance(value_type, type), f"value_type must be a type"
        if value_type is not None:
            if self.cache is not None:
                assert type(self.cache) == value_type, f"the type of cache must be {value_type}, not {type(self.cache)}"
            else:
                self.__value_type = value_type
        else:
            self.__value_type = None

    @property
    def cache(self):
        return self.__cache
    
    @cache.setter
    def cache(self, cache):
        if self.value_type is not None and cache is not None:
            assert isinstance(cache, self.value_type), f"the type of cache must be {self.value_type}, not {type(cache)}"
        self.__cache = cache

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
        ".txt":  [read_file_as_str, write_str_to_file, str],
        ".png":  [cv2.imread, cv2.imwrite, None],
        ".jpg":  [cv2.imread, cv2.imwrite, None],
        ".jpeg": [cv2.imread, cv2.imwrite, None],
        ".bmp":  [cv2.imread, cv2.imwrite, None],
        ".tif":  [cv2.imread, cv2.imwrite, None],
    }

    def __init__(self, cluster:FCT, sub_dir:str, corename:str, suffix:str, * ,
                 prefix:str = "", appendnames:list[str] = None,  # type: ignore
                 prefix_joiner:str = '', appendnames_joiner:str = '',
                 data_path = "",
                 read_func:Callable = None, write_func:Callable = None, 
                 cache = None, value_type:Callable = None) -> None: #type: ignore
        if hasattr(self, "_inited"):
            return
        super().__init__()
        appendnames = appendnames if appendnames is not None else ['']
        if suffix[0] != '.':
            suffix = '.' + suffix
        
        self.cluster:FCT = cluster

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

        self.cache_proxy:CacheProxy = CacheProxy(cache, value_type)

        self._inited = True

    def __hash__(self):
        return hash(self.get_path())
    
    def __eq__(self, o: object) -> bool:
        if isinstance(o, FilesHandle):
            return hash(self) == hash(o)
        else:
            return False

    def __setattr__(self, name, value):
        if hasattr(self, "_inited"):
            raise AttributeError(f"FilesHandle is immutable, you can't change its attribute")
        return super().__setattr__(name, value)

    ##IO##        
    @property
    def closed(self):
        self.cluster.closed

    @property
    def readonly(self):
        self.cluster.readonly

    @property
    def overwrite_forbidden(self):
        self.cluster.overwrite_forbidden
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
        if not self.closed:
            if not self.readonly:
                return self.cache_proxy.cache
            else:
                return copy.copy(self.cache_proxy.cache)
        else:
            return None

    def set_cache(self, cache):
        if not self.closed and not self.readonly:
            self.cache_proxy.cache = cache

    def erase_cache(self):
        if not self.closed and not self.readonly:
            self.cache_proxy.cache = None    

    def set_synced(self, synced:bool = True):
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
        if file.startswith('.'):
            suffix = file
        else:
            suffix = os.path.splitext(file)[1]
        if suffix in cls.DEFAULT_FILE_TYPE:
            _read_func, _write_func, _value_type = cls.DEFAULT_FILE_TYPE[suffix]
            read_func = read_func if read_func is not None else _read_func
            write_func = write_func if write_func is not None else _write_func
            value_type = value_type if value_type is not None else _value_type
        else:
            warnings.warn(f"can't find default file type for {suffix}, use str as default value_type", ClusterNotRecommendWarning)

        return read_func, write_func, value_type

    def get_key(self):
        return self.cluster._MemoryData._reverse_dict[self] # type: ignore

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
        return f"BinDict({super().__repr__()})"

class DataMapping(IOStatusManager, ABC, Generic[_VT, DMT, VDMT]):
    _registry:dict = {}

    _DMT = TypeVar('_DMT', bound="DataMapping")
    _VDMT = TypeVar('_VDMT')

    MULTI_FILES = False
    MEMORY_DATA_FILE = ".datamap"

    KEY_TYPE = object
    file_handle_type:type[FilesHandle] = FilesHandle

    load_memory_func:Callable[[str], dict]         = deserialize_object
    save_memory_func:Callable[[str, dict], None]   = serialize_object

    #############
    def __new__(cls, top_directory:str, name: str, *args, **kwargs):
        if cls is DataMapping:
            raise TypeError("DataMapping cannot be instantiated")
        # single instance
        obj = super().__new__(cls)
        obj.init_identity(top_directory, name)
        if obj.identity_string() in cls._registry:
            return cls._registry[obj.identity_string()]
        else:
            cls._registry[obj.identity_string()] = obj
            return obj

    @staticmethod
    def parse_identity_string(identity_string:str):
        cls_name, directory_name = identity_string.split(':')
        directory, mapping_name = os.path.split(directory_name)
        return cls_name, directory, mapping_name

    @classmethod
    def gen_identity_string(cls, directory, mapping_name):
        return f"{cls.__name__}:{os.path.normpath(os.path.join(directory, mapping_name))}"

    def identity_string(self):
        return self.gen_identity_string(self.top_directory, self.mapping_name)

    def init_identity(self, top_directory:str, name: str):
        self._unfinished_operation = 0
        self.__directory = top_directory
        self.mapping_name = name        

    def key_identity_string(self):
        return f"{self.__class__.__name__}:{self.name}"        
    #############

    def __init__(self, top_directory:Union[str, "DatasetNode"], name: str, *args, **kwargs) -> None:
        '''
        Initialize the data cluster with the provided top_directory, name, and registration flag.
        '''
        IOStatusManager.__init__(self, name)
        Generic.__init__(self)
        self._MemoryData:dict[int, _VT] = {}

        self.cache_priority     = True
        self.strict_priority_mode    = False
        self.write_synchronous  = False
        self.changed_since_opening = False

        if os.path.exists(self.data_path):
            self.open()  # Opens the cluster for operation.
        else:
            self.close()
    
    @property
    def top_directory(self):
        return self.__directory

    @top_directory.setter
    def top_directory(self, value):
        return None

    def make_path(self):
        if not os.path.exists(self.data_path):
            if '.' in os.path.basename(self.data_path):
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
    def MemoryData_path(self):
        return os.path.join(self.top_directory, self.mapping_name, self.MEMORY_DATA_FILE)
    
    @property
    def data_path(self):
        return os.path.join(self.top_directory, self.mapping_name)

    def save_preprecess(self):
        return self._MemoryData

    def load_postprocess(self, data):
        return data

    def save(self):
        self.__class__.save_memory_func(self.MemoryData_path, self.save_preprecess())

    def load(self):
        if os.path.exists(self.MemoryData_path):
            self._MemoryData = self.load_postprocess(self.__class__.load_memory_func(self.MemoryData_path))
        else:
            self._MemoryData = self.load_postprocess({})
            self.rebuild()
            self.save()

    def sort(self):
        new_dict = dict(sorted(self._MemoryData.items(), key=lambda x:x[0])) # type: ignore
        self._MemoryData.clear()
        self._MemoryData.update(new_dict)    

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
    def merge_from(self, src:DMT, force = False) -> None:
        pass

    @abstractmethod
    def copy_from(self, src:DMT, cover = False, force = False) -> None:
        pass
    ####################

    #######################
    @property
    def data_num(self):
        return len(self)

    @property
    def data_i_upper(self):
        if self.KEY_TYPE != int:
            raise TypeError(f"the key_type of {self.__class__.__name__} is not int")
        return max(self.keys()) + 1 if len(self) > 0 else 0 # type: ignore
 
    @property
    def continuous(self):
        return self.data_num == self.data_i_upper

    def keys(self):
        return self._MemoryData.keys()

    def values(self) -> Generator[VDMT, Any, None]:
        def value_generator():
            for i in self._MemoryData.keys():
                yield self.read(i)
        return value_generator()
    
    def items(self):
        def items_generator():
            for i in self._MemoryData.keys():
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
        return len(self._MemoryData)
    ####################

    ### complex io ####
    def append(self, value:VDMT, force = False, **other_paras):
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        dst = self.data_i_upper
        with self.writer.allow_overwriting().valid(force):
            self.write(dst, value, **other_paras)

    def clear(self, force = False):
        with self.writer.allow_overwriting().valid(force):
            ### TODO
            for key in tqdm(list(self.keys()), desc=f"clear {self}"):
                self.remove(key)

    def make_continuous(self, force = False):
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        if self.continuous:
            return
        with self.writer.allow_overwriting().valid(force):
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
        with d1.writer:
            for array in d0: # type: ignore
                d1.append(array)
        print("append passed", time.time() - start)

        start = time.time()
        d1.clear(force=True)
        print("clear passed", time.time() - start)

        start = time.time()
        with d1.writer:
            d1.copy_from(d0, cover=True) # type: ignore
        print("copy_from passed", time.time() - start)

        with d1.writer:
            d1.remove(0)
        start = time.time()
        d1.make_continuous(True)
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
    def _FC_MemoryData(self)  -> BinDict[int, FHT]:
        return self.files_cluster._MemoryData # type: ignore

    @abstractmethod
    def get_FilesHandle(self, src, dst, value, **other_paras) -> tuple[FHT, FHT]:
        pass

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

    def format_value(self, value) -> Any:
        return value
    
    def inv_format_value(self, formatted_value) -> Any:
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

    def __call__(self, *, src = None, dst = None, value = None, **other_paras) -> Any:
        src_handle, dst_handle = self.get_FilesHandle(src = src, dst = dst, value = value, **other_paras)
        if not self.__io_raw:
            value = self.preprogress_value(value, **other_paras)
        rlt = self.io(src_handle, dst_handle, value)
        if not self.__io_raw:
            rlt = self.postprogress_value(rlt, **other_paras)
        self.progress_FilesHandle(src_handle, dst_handle, rlt, **other_paras)
        return rlt

    def gather_mutil_results(self, results:list):
        raise NotImplementedError

    def split_value_as_mutil(self, *core_values):
        raise NotImplementedError

    def check_src(self, src):
        if not isinstance(src, self.key_type):
            return False
        if src not in self.files_cluster.keys():
            return False
        return True
        
    def check_dst(self, dst):
        if not isinstance(dst, self.key_type):
            return False
        return True

    def check_value(self, value: Any):
        return True

    def is_overwriting(self, dst:int):
        if dst in self.files_cluster.keys():
            fh = self.files_cluster.query_fileshandle(dst)
            return not fh.empty
        return False

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

    IS_ELEM = False
    KEY_TYPE = int

    #############
    def init_identity(self, dataset_node:Union[str, "DatasetNode"], name: str):
        if isinstance(dataset_node, str):
            self._unfinished_operation = 0
            self.__dataset_node:DSNT = None # type: ignore 
            self.__directory = dataset_node
        elif isinstance(dataset_node, DatasetNode):
            self._unfinished_operation = self.__dataset_node._unfinished_operation
            self.__dataset_node:DSNT = dataset_node # type: ignore
        else:
            raise TypeError(f"dataset_node must be str or DatasetNode, not {type(dataset_node)}")
        self.mapping_name = name        
    #############

    def __init__(self, dataset_node: Union[str, "DatasetNode"], name: str, *args, **kwargs) -> None:
        super().__init__(dataset_node, name)

        self._MemoryData:BinDict[int, FHT] = self._MemoryData
        self._unfinished = self.mark_exist()
        self._unfinished_operation = 0

        self.init_io_metas()
        self.register_to_dataset()

    @property
    def top_directory(self) -> str:
        if self.__dataset_node is not None:
            return self.__dataset_node.top_directory
        else:
            return self.__directory

    @top_directory.setter
    def top_directory(self, value):
        return None

    @property
    def dataset_node(self):
        return self.__dataset_node
    
    @dataset_node.setter
    def dataset_node(self, dataset_node):
        if self.__dataset_node is not None:
            self.unregister_from_dataset()
        self.__dataset_node = dataset_node
        self.register_to_dataset()

    @property
    def registerd(self):
        return self.__dataset_node is not None and \
            self.identity_string() in self.__dataset_node.cluster_map

    def register_to_dataset(self):
        if self.__dataset_node is not None:
            self.__dataset_node.cluster_map.add_cluster(self)

    def unregister_from_dataset(self):
        if self.identity_string() in self.__dataset_node.cluster_map:
            self.__dataset_node.cluster_map.remove_cluster(self)

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
                    yield self._MemoryData[i]
            return g()
        else:
            return self._MemoryData[data_i]
    
    @abstractmethod
    def create_fileshandle(self, src, dst, value, **other_paras) -> FHT:
        pass

    def format_corename(self, data_i:int):
        return None
    
    def deformat_corename(self, corename:str):
        return None
    
    def _set_fileshandle(self, data_i, fileshandle:FHT):
        if fileshandle not in self._MemoryData._reverse_dict:
            self._MemoryData[data_i] = fileshandle

    def _pop_fileshandle(self, data_i):
        if data_i in self._MemoryData.keys():
            return self._MemoryData.pop(data_i)
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
        
        def wrapper(*, src:int = None, dst:int = None, value = None, **other_paras): # type: ignore
            nonlocal self, log_type, warning_info
            src = self.cvt_key(src)
            dst = self.cvt_key(dst)
            rlt = None
            with self._IOContext(self, force, force, force).valid(force): 
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
                        warnings.warn(f"{self.__class__.__name__}:{self.mapping_name} \
                                    is not allowed to overwitre, any write operation will not be executed.",
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
                            if overwrited and log_type == self.LOG_APPEND:
                                log_type = self.LOG_CHANGE
                            self.log_to_mark_file(log_type, src, dst, value)
                
                if io_error:
                    io_type_name = ["READ", "APPEND", "REMOVE", "CHANGE", "MOVE", "OPERATION"]
                    warnings.warn(f"{self.__class__.__name__}:{self.mapping_name} \
                        {io_type_name[log_type]}: src:{src}, dst:{dst}, value:{value} failed:\
                        {warning_info}", ClusterIONotExecutedWarning)

            return rlt
        return wrapper

    _FCT = TypeVar('_FCT', bound="FilesCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=FilesHandle)

    class _read(IOMeta[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value, **other_paras):
            if src in self.files_cluster.keys():
                return self._FC_MemoryData[src], None
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
        LOG_TYPE = IOStatusManager.LOG_APPEND

        def get_FilesHandle(self, src, dst, value,  **other_paras):
            if dst not in self.files_cluster.keys():
                fh:FilesHandle = self.files_cluster.create_fileshandle(src, dst, value, **other_paras)
                self.files_cluster._set_fileshandle(dst, fh)
            return None, self._FC_MemoryData[dst]

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
            src_handle:FilesHandle = self._FC_MemoryData.pop(src) # type: ignore
            if dst not in self.files_cluster.keys():
                dst_handle = self.files_cluster.file_handle_type.from_fileshandle(self.files_cluster, src_handle, 
                                                                                  corename= self.files_cluster.format_corename(dst))
                self.files_cluster._set_fileshandle(dst, dst_handle)
            dst_handle = self._FC_MemoryData[dst]
                
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
            if dst not in self.files_cluster.keys():
                fh = self.files_cluster.create_fileshandle(src, dst, value, **other_paras)
            else:
                fh = self._FC_MemoryData[dst]
            return None, fh

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value=None) -> Any:
            dst_file_handle.erase_cache()

        def cvt_to_core_paras(self, 
                                src_file_handle: FilesHandle, 
                                dst_file_handle: FilesHandle, 
                                value, 
                                **other_paras) -> tuple:
            return (dst_file_handle.get_path(), )

        def progress_FilesHandle(self, src_file_handle, dst_file_handle:FilesHandle, value, **other_paras):
            if dst_file_handle.empty:
                key = dst_file_handle.get_key()
                self._FC_MemoryData.pop(key)

    class _paste_file(IOMeta[_FCT, _VDMT, _FHT]):
        READ = False
        W_SYNC = True
        PATH_EXISTS_REQUIRED = True
        LOG_TYPE = IOStatusManager.LOG_APPEND

        def __init__(self, files_cluster) -> None:
            super().__init__(files_cluster)
            self.core_func = shutil.copy

        def get_FilesHandle(self, src, dst, value, **other_paras):
            src_handle:FilesHandle = value
            if dst not in self.files_cluster.keys():
                dst_handle = self.files_cluster.file_handle_type.from_fileshandle(self.files_cluster, src_handle, 
                                                                                  corename= self.files_cluster.format_corename(dst))
                self.files_cluster._set_fileshandle(dst, dst_handle)
            dst_handle = self._FC_MemoryData[dst]
            return None, self._FC_MemoryData[dst]

        def io_cache(self, src_file_handle, dst_file_handle:FilesHandle, value:FilesHandle=None) -> Any: # type: ignore
            pass

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
            dst_handle = self.files_cluster.file_handle_type.from_fileshandle(self.files_cluster, src_handle, sub_dir=value)
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

    def read(self, src:int, *, force = False, **other_paras) -> VDMT:
        return self.io_decorator(self.read_meta, force)(src = src, **other_paras) # type: ignore

    def write(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.write_meta, force)(dst = dst, value = value, **other_paras)
    
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.modify_key_meta, force)(src = src, dst = dst, **other_paras)
    
    def remove(self, dst:int, remove_both = False, *, force = False, **other_paras) -> None:
        if remove_both:
            self.remove_meta._set_ctrl_flag(write_synchronous=True)
        rlt = self.io_decorator(self.remove_meta, force)(dst = dst, **other_paras)
        self.read_meta._clear_ctrl_flag()
        return rlt
        
    def paste_file(self, dst:int, file_handler:FilesHandle, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.paste_file_meta, force)(dst = dst, value = file_handler, **other_paras)

    def change_dir(self, dst:int, new_dir_name, *, force = False, **other_paras) -> None:
        return self.io_decorator(self.change_dir_meta, force)(dst = dst, value = new_dir_name, **other_paras)

    def merge_from(self, src_data_map:FCT, force = False):
        def merge_func(src_data_map:FCT, dst_data_map:FCT, data_i:int, data_i_upper:int):
            dst_data_map.paste_file(
                data_i_upper,
                src_data_map.query_fileshandle(data_i)) # type: ignore
        
        assert self.KEY_TYPE == int, f"the key_type of {self.__class__.__name__} is not int"
        assert src_data_map.KEY_TYPE == int, f"the key_type of {src_data_map.__class__.__name__} is not int"
        assert type(src_data_map) == type(self), f"can't merge {type(src_data_map)} to {type(self)}"
        # assert self.opened and self.writable, f"{self.__class__.__name__} is not writable"
        assert src_data_map.opened, f"{src_data_map.__class__.__name__} is not opened"

        # if self.continuous:
        #     return

        with self.writer.valid(force):
            for data_i in tqdm(src_data_map.keys(), desc=f"merge {src_data_map} to {self}", total=len(src_data_map)):
                merge_func(src_data_map, self, data_i, self.data_i_upper)

    def copy_from(self, src_data_map:FCT, cover = False, only_data_map = False, force = False):
        if only_data_map:
            if os.path.exists(self.MemoryData_path):
                if cover:
                    os.remove(self.MemoryData_path)
                    shutil.copyfile(src_data_map.MemoryData_path, self.MemoryData_path)
                    self.load()
                else:
                    raise IOError(f"{self.MemoryData_path} already exists")
        else:
            if os.path.exists(self.data_path) and len(self) > 0:
                if cover:
                    shutil.rmtree(self.data_path)
                    os.makedirs(self.data_path)
                    self._MemoryData.clear()
                    self.save()
                else:
                    raise IOError(f"{self.data_path} already exists")
            self.merge_from(src_data_map, force) 
    
    def clear(self, force = False, clear_both = True):
        if clear_both:
            with self.remove_meta._set_ctrl_flag(write_synchronous=True):
                super().clear(force)
        else:
            super().clear(force)

    def cache_to_file(self, data_i:int = None, *, force = False, **other_paras):
        rlt = False
        self.write_meta._set_ctrl_flag(cache_priority=False, strict_priority_mode=True, 
                                       write_synchronous=False, io_raw=True)
        data_i_list = [data_i] if isinstance(data_i, int) else self.keys()
        for data_i in data_i_list:
            fh = self._MemoryData[data_i]
            if fh.synced or not fh.has_cache:
                continue
            value = fh.cache
            rlt = self.write(data_i, value, force = force, **other_paras)
            self.query_fileshandle(data_i).set_synced(True)
        self.write_meta._clear_ctrl_flag()
        return rlt

    def file_to_cache(self, data_i:int = None, *, save = True, force = False, **other_paras):
        rlt = False
        data_i_list = [data_i] if isinstance(data_i, int) else self.keys()

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
            rlt = self.write(data_i, value, force = force, **other_paras)
            self.query_fileshandle(data_i).set_synced(True)

        self.write_meta._clear_ctrl_flag()
        self.read_meta._clear_ctrl_flag()

        if save:
            self.save()
        return rlt
    #### io metas END#####

    #### Memorydata operation ####
    def matching_path(self):
        paths:list[str] = []
        paths.extend(glob.glob(os.path.join(self.data_path, "**/*"), recursive=True))
        return paths

    def rebuild(self):
        paths:list[str] = self.matching_path()

        for path in paths:
            fh = self.file_handle_type.from_path(self, path)
            data_i = self.deformat_corename(fh.corename)
            data_i = data_i if data_i is not None else self.data_i_upper
            if fh.all_file_exist:
                self._set_fileshandle(data_i, fh)
            else:
                self.paste_file(data_i, fh)

        for fh in list(self._MemoryData.values()):
            if fh.empty:
                self.remove(fh)

        self.sort()

    def save_preprecess(self):
        to_save_dict = {item[0]: item[1].as_dict() for item in self._MemoryData.items()}
        return to_save_dict
    
    def load_postprocess(self, data:dict):
        new_dict = {int(k): self.file_handle_type.from_dict(self, v) for k, v in data.items()}
        data_info_map = BinDict(new_dict)
        return data_info_map
    ###############


class _ClusterMap(dict[str, FilesCluster]):
    def __init__(self, dataset_node:"DatasetNode", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_node = dataset_node

    def __set_update(self):
        self.dataset_node.updated = True
        if self.dataset_node.inited:
            self.dataset_node.update_dataset()

    def __setitem__(self, __key: Any, __value: Any) -> None:
        self.__set_update()
        return super().__setitem__(__key, __value)
    
    def update(self, __m, **kwargs: Any) -> None:
        self.__set_update()
        return super().update(__m, **kwargs)

    def setdefault(self, __key: Any, __default: Any = ...) -> Any:
        self.__set_update()
        return super().setdefault(__key, __default)

    def search(self, __key: Any, return_index = False):
        return search_in_dict(self, __key, process_func=self._search_func) ### TODO
    
    def add_cluster(self, cluster:FilesCluster):
        cluster.dataset_node = self.dataset_node
        self[cluster.mapping_name] = cluster
        self.dataset_node._MemoryData.add_column(cluster.mapping_name, exist_ok=True)

    def remove_cluster(self, cluster:FilesCluster):
        if cluster.dataset_node == self.dataset_node:
            del self[cluster.mapping_name]
            self.dataset_node._MemoryData.remove_column(cluster.mapping_name, not_exist_ok=True)

    def get_keywords(self):
        keywords = []
        for cluster in self.values():
            keywords.append(cluster.mapping_name)
        return keywords

    @staticmethod
    def _search_func(indetity_string:str):
        cls_name, directory, name = DataMapping.parse_identity_string(indetity_string)
        return name 

class DatasetNode(DataMapping[dict[str, bool], DSNT, VDST], ABC, Generic[FCT, DSNT, VDST]):
   
    load_memory_func = Table.from_json
    save_memory_func = Table.to_json    
    
    def __init__(self, directory, parent:"DatasetNode" = None) -> None:
        super().__init__(directory, "")

        self.__inited = False # if the dataset has been inited   
        self.init_node_hook(parent)
        self.init_dataset_attr()
        self._MemoryData:Table[int, str, bool] = Table[int, str, bool](default_value_type=bool, row_name_type=int, col_name_type=str)
        self.init_clusters_hook()
        self.__inited = True # if the dataset has been inited

    def init_node_hook(self, parent):
        self.parent:DatasetNode = parent
        self.children:list[DatasetNode] = []
        self.move_node(parent)

    def init_dataset_attr(self):
        def is_directory_inside(base_dir, target_dir):
            base_dir:str = os.path.abspath(base_dir)
            target_dir:str = os.path.abspath(target_dir)
            return target_dir.startswith(base_dir)
        
        os.makedirs(self.top_directory, exist_ok=True) 
        self._updated = False
        self._unfinished_operation = 0
        if self.parent is not None:
            assert is_directory_inside(self.parent.top_directory, self.top_directory), f"{self.top_directory} is not inside {self.parent.top_directory}"
            self.name:str = os.path.relpath(self.top_directory, self.parent.top_directory)
        else:
            self.name:str = self.top_directory
        self.cluster_map = _ClusterMap[FCT](self)
 
    ##### IOstatus #####
    def open_hook(self):
        for obj in self.clusters:
            obj.open()

    def close_hook(self):
        for obj in self.clusters:
            obj.close()

    def readonly_hook(self):
        for obj in self.clusters:
            obj.set_readonly()

    def writable_hook(self):
        for obj in self.clusters:
            obj.set_writable()

    def stop_writing_hook(self):
        for obj in self.clusters:
            obj.stop_writing()

    def start_writing_hook(self):
        for obj in self.clusters:
            obj.start_writing()

    def set_overwrite_allowed_hook(self):
        for obj in self.clusters:
            obj.set_overwrite_allowed()

    def set_overwrite_forbidden_hook(self):
        for obj in self.clusters:
            obj.set_overwrite_forbidden()
    ####################

    ##### clusters #####
    def init_clusters_hook(self):
        unfinished = self.mark_exist()
        if unfinished:
            y:int = FilesCluster.choose_unfinished_operation(self)
            self._unfinished_operation = y        

        self._init_clusters()
        self.update_dataset()        

        self.load_overview()        
        if unfinished:
            self.process_unfinished()
            os.remove(self.get_writing_mark_file())

    @property
    def clusters(self) -> list[FCT]:
        clusters = list(self.cluster_map.values())
        return clusters

    @property
    def elem_clusters(self) -> list[FCT]:
        clusters = [x for x in self.clusters if x.IS_ELEM]
        return clusters

    @property
    def opened_clusters(self):
        clusters = [x for x in self.clusters if x.opened]
        return clusters

    @property
    def opened_elem_clusters(self):
        clusters = [x for x in self.elem_clusters if x.opened]
        return clusters
    
    def get_all_clusters(self, _type:Union[type, tuple[type]] = None, only_opened = False) -> _ClusterMap[FCT]:
        cluster_map = _ClusterMap[FilesCluster](self)
        cluster_map.update(self.cluster_map)

        for k, v in list(cluster_map.items()):
            if _type is not None:
                if not isinstance(v, _type):
                    cluster_map.pop(k)
            if only_opened:
                if v.is_closed():
                    cluster_map.pop(k)

        for child in self.children:
            cluster_map.update(child.get_all_clusters(_type, only_opened))

        return cluster_map
    
    def __setattr__(self, name, value):
        ### 同名变量赋值时，自动将原有对象解除注册
        if name in self.__dict__:
            obj = self.__getattribute__(name)
            if isinstance(obj, DatasetNode):
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
    def read(self, src:_KT):
        pass

    def write(self, dst:int, value:VDMT, *, force = False, **other_paras) -> None:
        pass
        
    def modify_key(self, src:int, dst:int, *, force = False, **other_paras) -> None:
        pass

    def remove(self, dst:int, *, force = False, **other_paras) -> None:
        pass

    def merge_from(self, src:DSNT, force = False) -> None:
        pass

    def copy_from(self, src:DSNT, force = False) -> None:
        pass
    ############

    ############
    def rebuild(self):
        if len(self.elem_clusters) > 0:
            rows = [x for x in range(self.data_i_upper)]
            cols = [x.key_identity_string() for x in self.elem_clusters]
            self._MemoryData = Table(rows, cols, bool, row_name_type=int, col_name_type=str)
            for data_i in tqdm(self._MemoryData.data, desc="initializing data frame"):
                self.calc_overview(data_i)

    def load(self):
        return super().load()

    def calc_overview(self, data_i):
        self._MemoryData.add_row(data_i, exist_ok=True)        
        for cluster in self.elem_clusters:
            self._MemoryData[data_i, cluster.key_identity_string()] = data_i in cluster.keys()

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