# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_items, dict_keys, dict_values
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
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
import copy

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Generator
from functools import partial


from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt, \
    serialize_object, deserialize_object, read_file_as_str, write_str_to_file
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta
from .IOAbstract import DataMapping, DatasetNode, IOMeta, _KT, _VT, DMT, VDMT, DSNT,\
    FilesHandle, \
    AmbiguousError, IOMetaParameterError, KeyNotFoundError, ClusterDataIOError, IOStatusWarning, ClusterIONotExecutedWarning, ClusterNotRecommendWarning,\
    FilesCluster,\
    parse_kw

FHT = TypeVar('FHT', bound=FilesHandle) # type of the files handle
UFC = TypeVar('UFC', bound="UnifiedFileCluster") # type of the unified file cluster
DFC = TypeVar('DFC', bound="DisunifiedFileCluster") # type of the disunified file cluster
DLC = TypeVar('DLC', bound="DictLikeCluster") # type of the dict-like cluster
VDMT = TypeVar('VDMT') # type of the value of data cluster

class UnifiedFilesHandle(FilesHandle[UFC, VDMT], Generic[UFC, VDMT]):
    def __init__(self, cluster:"UnifiedFileCluster", sub_dir:str, corename:str, suffix:str, * ,
                 prefix:str = "", appendnames:list[str] = None,  # type: ignore
                 prefix_joiner:str = '', appendnames_joiner:str = '',
                 data_path = "",
                 read_func:Callable[[str], Any] = None, write_func:Callable[[str, Any], None] = None,
                 cache = None, value_type:Callable = None) -> None: #type: ignore
        read_func  = cluster.read_func 
        write_func = cluster.write_func
        value_type = cluster.value_type if value_type is None else value_type
        super().__init__(cluster, sub_dir, corename, suffix, prefix=prefix, appendnames=appendnames,
                            prefix_joiner=prefix_joiner, appendnames_joiner=appendnames_joiner,
                            data_path=data_path,
                            read_func=read_func, write_func=write_func,
                            cache=cache, value_type=value_type)

class UnifiedFileCluster(FilesCluster[UnifiedFilesHandle, "UnifiedFileCluster", DSNT, VDMT], Generic[DSNT, VDMT]):
    file_handle_type = UnifiedFilesHandle
    def __init__(self, dataset_node:DSNT, name: str,
                 suffix:str = '.txt', *,
                 read_func:Callable[[str], VDMT] = None, 
                 write_func:Callable[[str, VDMT], None] = None, 
                 value_type:Callable = None,
                 filllen = 6, 
                 fillchar = '0',
                 alternate_suffix:list = None, 
                 **kwargs
                 ) -> None:
        self.suffix = suffix

        read_func, write_func, value_type = FilesHandle.try_get_default(suffix, read_func, write_func, value_type)
        self.read_func = read_func
        self.write_func = write_func
        self.value_type = value_type

        self.filllen = filllen
        self.fillchar = fillchar
        self.alternate_suffix = alternate_suffix if alternate_suffix is not None else []  
        self.cache_priority = True 
        super().__init__(dataset_node, name)

    #####
    def create_fileshandle(self, src, dst, value, sub_dir, **other_paras):
        if not self.MULTI_FILES:
            corename = self.format_corename(dst)
            fh = self.file_handle_type(self, sub_dir, corename, self.suffix)
            return fh
        else:
            raise NotImplementedError
    #####
    def format_corename(self, data_i: int):
        filllen = self.filllen
        fillchar = self.fillchar
        return f"{str(data_i).rjust(filllen, fillchar)}"
    
    def deformat_corename(self, corename: str):
        return int(corename)

    def matching_path(self):
        paths = []
        for suffix in self.alternate_suffix + [self.suffix]:
            paths.extend(glob.glob(os.path.join(self.data_path, "**/*" + suffix), recursive=True))
        return paths
        
    ############################
    def init_io_metas(self):
        '''
        init the io_metas of the data cluster
        '''
        super().init_io_metas()
        self.change_dir_meta:IOMeta["UnifiedFileCluster", VDMT] = self._change_dir(self)

    class _read(FilesCluster._read["UnifiedFileCluster", FilesCluster._VDMT, FilesCluster._FHT]):
        @property
        def core_func(self):
            return self.files_cluster.read_func
        
        @core_func.setter
        def core_func(self, func):
            pass

    class _write(FilesCluster._write["UnifiedFileCluster", FilesCluster._VDMT, FilesCluster._FHT]):
        def get_FilesHandle(self, src, dst, value, *, sub_dir = "", **other_paras):
            return super().get_FilesHandle(src, dst, value, sub_dir = sub_dir, **other_paras)
        
        @property
        def core_func(self):
            return self.files_cluster.write_func
        
        @core_func.setter
        def core_func(self, func):
            pass      

    class _paste_file(FilesCluster._paste_file["UnifiedFileCluster", FilesCluster._VDMT, FilesCluster._FHT]):
        def get_FilesHandle(self, src, dst, value:UnifiedFilesHandle, **other_paras):
            sub_dir = value.sub_dir
            return super().get_FilesHandle(src, dst, value, sub_dir = sub_dir, **other_paras)

    def clear(self, force=False, clear_both=True):
        super().clear(force, clear_both)
        for r, d, f in os.walk(self.data_path):
            for dir_ in d:
                if len(os.listdir(os.path.join(r, dir_))):
                    os.rmdir(os.path.join(r, dir_))

    def write(self, data_i:int, value:VDMT, sub_dir = "", *, force = False, **other_paras) -> None:
        return super().write(data_i, value, sub_dir=sub_dir, force=force, **other_paras)
    
    def append(self, value: VDMT, sub_dir = "", force=False, **other_paras):
        return super().append(value, force, sub_dir = sub_dir, **other_paras)
    #####################

    #######################
    ########################

    @staticmethod
    def ElementsTest():
        top_dir = os.path.join(os.getcwd(), "ElementsTest")
        cv2.imwrite(os.path.join(top_dir, "000000.png"), np.random.random((640, 480, 3))*255)
        e0 = UnifiedFileCluster(top_dir, "000000", suffix=".jpg", alternate_suffix=[".png"])
        e1 = UnifiedFileCluster(top_dir, "000001", suffix=".jpg", alternate_suffix=[".png"])

        e0.clear(force=True)
        e1.clear(force=True)

        start = time.time()
        with e0.writer:
            for i in range(5):    
                e0.append((np.random.random((640, 480, 3))*255).astype(np.uint8), "sub0")
        print(time.time() - start)

        e0.cache_priority = False
        with e0.writer:
            for i in range(5):    
                e0.append((np.random.random((640, 480, 3))*255).astype(np.uint8), "sub1")

        for fh in e0.query_fileshandle(0, 'end'):
            print(fh)

        with e0.writer.allow_overwriting():
            e0.cache_to_file()
            e0.file_to_cache()

        os.remove(e0.MemoryData_path)
        e0.close()
        e0.open()

        print()
        for fh in e0.query_fileshandle(0, 'end'):
            print(fh)

        start = time.time()
        with e1.writer:
            for array in e0:
                e1.append(array)
        print(time.time() - start)
        print()
        for fh in e1.query_fileshandle(0, 'end'):
            print(fh)

        e1.file_to_cache()
        e1.clear(force=True)

        start = time.time()
        with e1.writer:
            e1.copy_from(e0, cover=True)
        print(time.time() - start)

        with e1.writer:
            e1.remove(0, remove_both=True)
            e1.remove(5, remove_both=True)
        e1.make_continuous(True)

        print()
        for fh in e1.query_fileshandle(0, 'end'):
            print(fh)

        e0.clear(force=True)
        e1.clear(force=True)

class DisunifiedFilesHandle(FilesHandle[DFC, VDMT], Generic[DFC, VDMT]):
    _instances = {}
    def __new__(cls, *args, **kwargs) -> None:
        instance = super().__new__(cls)
        super().__init__(instance, *args, **kwargs)
        file_id_str = instance.__repr__()
        if file_id_str not in cls._instances:
            cls._instances[file_id_str] = instance
        return cls._instances[file_id_str]
        
    @property
    def file(self):
        return self.get_name()
    
    @file.setter
    def file(self, file):
        corename, suffix = os.path.splitext(os.path.basename(file))
        self.corename = corename
        self.suffix = suffix


    @classmethod
    def from_name(cls, cluster:"DisunifiedFileCluster", filename:Union[str, list[str]], *,
                prefix_joiner:str = '', appendnames_joiner:str = '', 
                read_func:Callable = None, write_func:Callable = None, 
                cache = None, value_type:Callable = None,  #type: ignore
                _extract_corename_func:Callable[[str], tuple[str, str, str, str, str]] = None):
        read_func, write_func, value_type = cls.try_get_default(filename, read_func, write_func, value_type)
        return super().from_name(cluster, filename,
                                    prefix_joiner=prefix_joiner, appendnames_joiner=appendnames_joiner, 
                                    read_func=read_func, write_func=write_func, 
                                    cache=cache, value_type=value_type, 
                                    _extract_corename_func=_extract_corename_func)

class DisunifiedFileCluster(FilesCluster[DisunifiedFilesHandle, "DisunifiedFileCluster", DSNT, DisunifiedFilesHandle], Generic[DSNT]):
    file_handle_type = DisunifiedFilesHandle

    #####
    def create_fileshandle(self, src:int, dst:int, value:Any, **other_paras):
        if not self.MULTI_FILES:
            if isinstance(value, DisunifiedFilesHandle):
                fh = self.file_handle_type.from_fileshandle(self, value)
                return fh
            else:
                raise NotImplementedError(f"can't create fileshandle")
        else:
            raise NotImplementedError
    #####

    def add_default_file(self, filename):
        suffix = os.path.splitext(filename)[-1]
        assert suffix in FilesHandle.DEFAULT_FILE_TYPE, f"suffix {suffix} is not supported"
        fh = self.file_handle_type.from_name(self, filename)
        self._set_fileshandle(self.data_i_upper, fh)

    def cvt_key(self, key:Union[int, str, DisunifiedFilesHandle]):
        self._MemoryData
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            if '.' in key:
                idx = list([f.file for f in self._MemoryData.values()]).index(key)
            else:
                idx = list([f.file for f in self._MemoryData.values()]).index(key)
            return list(self.keys())[idx]
        elif isinstance(key, DisunifiedFilesHandle):
            idx = list(self._MemoryData.values()).index(key)
            return list(self.keys())[idx]
        
    class _read(FilesCluster._read["DisunifiedFileCluster", FilesCluster._VDMT, FilesCluster._FHT]):
        def get_file_core_func(self, src_file_handle: DisunifiedFilesHandle, dst_file_handle: DisunifiedFilesHandle, value) -> Callable[..., Any]:
            return src_file_handle.read_func
        
    class _write(FilesCluster._write["DisunifiedFileCluster", FilesCluster._VDMT, FilesCluster._FHT]):
        def get_file_core_func(self, src_file_handle: DisunifiedFilesHandle, dst_file_handle: DisunifiedFilesHandle, value) -> Callable[..., Any]:
            return dst_file_handle.write_func
        
    @staticmethod
    def Test():
        def print_all_fh(fc:DisunifiedFileCluster):
            print()
            for fh in fc.query_fileshandle(0, 'end'):
                print(fh)

        top_dir = os.path.join(os.getcwd(), "DisunifiedFileClusterTest")

        d0 = DisunifiedFileCluster(top_dir, "000000")
        d1 = DisunifiedFileCluster(top_dir, "000001")

        d0.clear(force=True)
        d1.clear(force=True)

        d0.add_default_file("f0.npy")
        d0.add_default_file("f1.json")
        d0.add_default_file("f2.txt")
        d0.add_default_file("f3.pkl")
        d0.add_default_file("f4.png")
        d0.add_default_file("f0.npy")

        print_all_fh(d0)

        with d0.writer:
            d0.write(0, np.array([1, 2, 3]))
            d0.write(1, {"a": 1, "b": 2})
            d0.write(2, "hello world")
            d0.write(3, {"a": 1, "b": 2})
            d0.write(4, np.random.random((100, 100, 3)).astype(np.uint8))
        
        print_all_fh(d0)
        
        # TODO synced 控制逻辑？ copy_from 为什么不能拷贝cache
        d1.copy_from(d0, cover=True, force=True)
        print_all_fh(d1)

        d1.file_to_cache(force=True)
        print_all_fh(d1)
        d1.cache_to_file(force=True)
        print_all_fh(d1)

        d1.rebuild()

        with d1.writer.allow_overwriting():
            d1.remove(0, remove_both=True)
        d1.make_continuous(True)

        print_all_fh(d1)

        d0.clear(force=True)
        d1.clear(force=True)    

class DictLikeFilesHandle(DisunifiedFilesHandle[DLC, dict[int, Any]], Generic[DLC]):
    def get_cache_at(self, item_i):
        return self.cache[item_i]
    
    def set_cache_at(self, item_i, value):
        if self.cache_proxy.cache_io_mode == 'w':
            self.cache[item_i] = value

class DictLikeCluster(DisunifiedFileCluster):
    ### IO ###
    def get_item(self, item_i:Union[int, Iterable[int]]) -> Union[dict, list[dict]]:
        if isinstance(item_i, int):
            return_list = False
            item_i = [item_i]
        elif isinstance(item_i, Iterable):
            return_list = True
        else:
            raise TypeError(f"item_i should be int or Iterable[int], not {type(item_i)}")
        
        rlt = [] # save the return value
        item_i = [item_i] if isinstance(item_i, int) else item_i
        assert isinstance(item_i, Iterable), f"item_i should be int or Iterable[int], not {type(item_i)}"
        for data_i, dict_fh in self._MemoryData.items():
            dict_ = self.read(data_i) # read the dict
            rlt_dict = {}
            for ii in item_i:
                rlt_dict[data_i] = dict_[ii]
        
        if return_list:
            return rlt
        else:
            return rlt[0]
        
    def set_item(self, item_i:Union[int, Iterable[int]], value) -> Union[dict, list[dict]]:
        if isinstance(item_i, int):
            return_list = False
            item_i = [item_i]
        elif isinstance(item_i, Iterable):
            return_list = True
        else:
            raise TypeError(f"item_i should be int or Iterable[int], not {type(item_i)}")
        
        rlt = []
        item_i = [item_i] if isinstance(item_i, int) else item_i
        assert isinstance(item_i, Iterable), f"item_i should be int or Iterable[int], not {type(item_i)}"
        for data_i, dict_fh in self._MemoryData.items():
            dict_ = self.read(data_i)
            for ii in item_i:
                dict_[ii] = value
            self.write(data_i, dict_)
        
        if return_list:
            return rlt
        else:
            return rlt[0]