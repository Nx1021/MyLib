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


# from posture_6d.data.IOAbstract import DatasetNode


from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt, \
    serialize_object, deserialize_object, read_file_as_str, write_str_to_file
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta
from .IOAbstract import DataMapping, DatasetNode, IOMeta, BinDict, _KT, _VT, DMT, VDMT, DSNT,\
    FilesHandle, CacheProxy, \
    AmbiguousError, IOMetaParameterError, KeyNotFoundError, ClusterDataIOError, DataMapExistError, \
        IOStatusWarning, ClusterIONotExecutedWarning, ClusterNotRecommendWarning,\
    FilesCluster,\
    parse_kw

FHT = TypeVar('FHT', bound=FilesHandle) # type of the files handle
UFC = TypeVar('UFC', bound="UnifiedFileCluster") # type of the unified file cluster
DFC = TypeVar('DFC', bound="DisunifiedFileCluster") # type of the disunified file cluster
DLC = TypeVar('DLC', bound="DictLikeCluster") # type of the dict-like cluster
UFH = TypeVar('UFH', bound="UnifiedFilesHandle") # type of the unified files handle
DFH = TypeVar('DFH', bound="DisunifiedFilesHandle") # type of the disunified files handle
DLFH = TypeVar('DLFH', bound="DictLikeHandle") # type of the dict-like files handle
VDMT = TypeVar('VDMT') # type of the value of data cluster
IADC = TypeVar('IADC', bound="IntArrayDictCluster") # type of the input argument data cluster

class UnifiedFilesHandle(FilesHandle[UFC, VDMT], Generic[UFC, VDMT]):
    def init_input_hook(self, *,
                        read_func = None, write_func = None, value_type = None, #type: ignore
                        **kwargs):
        read_func  = self.cluster.read_func 
        write_func = self.cluster.write_func
        value_type = self.cluster.value_type if value_type is None else value_type
        return super().init_input_hook(read_func=read_func, write_func=write_func, value_type=value_type, **kwargs)

class UnifiedFileCluster(FilesCluster[UFH, UFC, DSNT, VDMT], Generic[UFH, UFC, DSNT, VDMT]):
    _IS_ELEM = True
    FILESHANDLE_TYPE = UnifiedFilesHandle
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
        super().__init__(dataset_node, name)
        self.cache_priority = False 

    #####
    def create_fileshandle(self, src, dst, value, * ,sub_dir = "", **other_paras):
        if not self.MULTI_FILES:
            corename = self.format_corename(dst)
            fh = self.FILESHANDLE_TYPE(self, sub_dir, corename, self.suffix)
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
    _FCT = TypeVar('_FCT', bound="UnifiedFileCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=UnifiedFilesHandle)

    def init_io_metas(self):
        '''
        init the io_metas of the data cluster
        '''
        super().init_io_metas()
        self.change_dir_meta:IOMeta["UnifiedFileCluster", VDMT] = self._change_dir(self)

    class _read(FilesCluster._read[_FCT, _VDMT, _FHT]):
        @property
        def core_func(self):
            return self.files_cluster.read_func
        
        @core_func.setter
        def core_func(self, func):
            pass

    class _write(FilesCluster._write[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value, *, sub_dir = "", **other_paras):
            return super().get_FilesHandle(src, dst, value, sub_dir = sub_dir, **other_paras)
        
        @property
        def core_func(self):
            return self.files_cluster.write_func
        
        @core_func.setter
        def core_func(self, func):
            pass      

    class _paste_file(FilesCluster._paste_file[_FCT, _VDMT, _FHT]):
        def get_FilesHandle(self, src, dst, value:UnifiedFilesHandle, **other_paras):
            sub_dir = value.sub_dir
            return super().get_FilesHandle(src, dst, value, sub_dir = sub_dir, **other_paras)

    def clear(self, force=False, clear_both=True):
        super().clear(force = force, clear_both=clear_both)


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
        with e0.get_writer():
            for i in range(5):    
                e0.append((np.random.random((640, 480, 3))*255).astype(np.uint8), "sub0")
        print(time.time() - start)

        e0.cache_priority = False
        with e0.get_writer():
            for i in range(5):    
                e0.append((np.random.random((640, 480, 3))*255).astype(np.uint8), "sub1")

        for fh in e0.query_fileshandle(0, 'end'):
            print(fh)

        with e0.get_writer().allow_overwriting():
            e0.cache_to_file()
            e0.file_to_cache()

        os.remove(e0.MemoryData_path)
        e0.close()
        e0.open()

        print()
        for fh in e0.query_fileshandle(0, 'end'):
            print(fh)

        start = time.time()
        with e1.get_writer():
            for array in e0:
                e1.append(array)
        print(time.time() - start)
        print()
        for fh in e1.query_fileshandle(0, 'end'):
            print(fh)

        e1.file_to_cache()
        e1.clear(force=True)

        start = time.time()
        with e1.get_writer():
            e1.copy_from(e0, cover=True)
        print(time.time() - start)

        with e1.get_writer():
            e1.remove(0, remove_both=True)
            e1.remove(5, remove_both=True)
        e1.make_continuous(True)

        print()
        for fh in e1.query_fileshandle(0, 'end'):
            print(fh)

        e0.clear(force=True)
        e1.clear(force=True)

class DisunifiedFilesHandle(FilesHandle[DFC, VDMT], Generic[DFC, VDMT]):
    # _instances = {}
    # def __new__(cls, *args, **kwargs) -> None:
    #     instance = super().__new__(cls)
    #     super().__init__(instance, *args, **kwargs)
    #     file_id_str = instance.__repr__()
    #     if file_id_str not in cls._instances:
    #         cls._instances[file_id_str] = instance
    #     return cls._instances[file_id_str]
        
    def init_input_hook(self, *, suffix:str, read_func, write_func, value_type, **kw):
        read_func, write_func, value_type = self.try_get_default(suffix, read_func, write_func, value_type)
        return super().init_input_hook(suffix = suffix, read_func=read_func, write_func=write_func, value_type=value_type, **kw)

    @property
    def file(self):
        return self.get_name()
    
    @file.setter
    def file(self, file):
        corename, suffix = os.path.splitext(os.path.basename(file))
        self.corename = corename
        self.suffix = suffix

class DisunifiedFileCluster(FilesCluster[DFH, DFC, DSNT, VDMT], Generic[DFH, DFC, DSNT, VDMT]):
    FILESHANDLE_TYPE = DisunifiedFilesHandle

    #####
    def create_fileshandle(self, src:int, dst:int, value:Any, **other_paras):
        if not self.MULTI_FILES:
            if isinstance(value, DisunifiedFilesHandle):
                fh = self.FILESHANDLE_TYPE.from_fileshandle(self, value)
                return fh
            else:
                fh = self.FILESHANDLE_TYPE.from_name(self, "_t.dfhtemp")
                return fh
        else:
            raise NotImplementedError
    #####

    @property
    def file_names(self):
        return [fh.file for fh in self.MemoryData.values()]

    def add_default_file(self, filename):
        suffix = os.path.splitext(filename)[-1]
        assert suffix in FilesHandle.DEFAULT_FILE_TYPE, f"suffix {suffix} is not supported"
        fh = self.FILESHANDLE_TYPE.from_name(self, filename)
        self._set_fileshandle(self.data_i_upper, fh)

    def cvt_key(self, key:Union[int, str, DisunifiedFilesHandle]):
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            if '.' in key:
                idx = list([f.file for f in self.MemoryData.values()]).index(key)
            else:
                idx = list([f.file for f in self.MemoryData.values()]).index(key)
            return list(self.keys())[idx]
        elif isinstance(key, DisunifiedFilesHandle):
            idx = list(self.MemoryData.values()).index(key)
            return list(self.keys())[idx]
        
    _FCT = TypeVar('_FCT', bound="DisunifiedFileCluster")
    _VDMT = TypeVar('_VDMT')
    _FHT = TypeVar('_FHT', bound=DisunifiedFilesHandle)

    class _read(FilesCluster._read[_FCT, _VDMT, _FHT]):
        def get_file_core_func(self, src_file_handle: DisunifiedFilesHandle, dst_file_handle: DisunifiedFilesHandle, value) -> Callable[..., Any]:
            return src_file_handle.read_func
        
    class _write(FilesCluster._write[_FCT, _VDMT, _FHT]):
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

        with d0.get_writer():
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

        with d1.get_writer().allow_overwriting():
            d1.remove(0, remove_both=True)
        d1.make_continuous(True)

        print_all_fh(d1)

        d0.clear(force=True)
        d1.clear(force=True)    

class DictLikeHandle(DisunifiedFilesHandle[DLC, dict[int, Any]], Generic[DLC]):
    LOAD_CACHE_ON_INIT = True

    def init_input_hook(self, *, value_type, **kw):
        return super().init_input_hook(value_type=dict, **kw)

    # def io_cache_at_wrapper(self, func:Callable, *, elem_i = None, value=None, is_read = False):
    #     io_error = False
    #     if self.is_closed:
    #         warnings.warn(f"can't set or get cache at {elem_i}", IOStatusWarning)
    #         io_error = True
    #     if  not is_read and self.is_readonly:
    #         warnings.warn(f"can't set cache at {elem_i}", IOStatusWarning)
    #         io_error = True
    #     if elem_i in self.cache and not is_read and self.overwrite_forbidden:
    #         warnings.warn(f"overwrite forbidden, can't set cache at {elem_i}", IOStatusWarning)
    #         io_error = True

    #     if func.__name__ == "__getitem__":
    #         return func(elem_i) if not io_error else None
    #     elif func.__name__ == "__setitem__":
    #         if io_error:
    #             return False
    #         else:
    #             func(elem_i, value)
    #             return True
    #     elif func.__name__ == "pop":
    #         return func(elem_i) if not io_error else None

    def get_cache_at(self, elem_i) -> Union[Any, None]:
        return self.cache[elem_i]
    
    def set_cache_at(self, elem_i, value) -> bool:
        return self.cache.__setitem__(elem_i, value)

    def pop_cache_at(self, elem_i) -> Union[Any, None]:
        return self.cache.pop(elem_i)
    
    def sort_cache(self):
        self.cache = dict(sorted(self.cache.items()))

    @property
    def elem_num(self):
        return len(self.cache)
    
    @property
    def elem_i_upper(self):
        return max(self.cache.keys(), default=-1) + 1
    
    @property
    def elem_continuous(self):
        return self.elem_i_upper == self.elem_num
        
    def erase_cache(self):
        if not self.is_closed and not self.is_readonly:
            self.cache.clear()

    @property
    def has_cache(self):
        return len(self.cache) > 0

class DictLikeCluster(DisunifiedFileCluster[DLFH, DLC, DSNT, VDMT], Generic[DLFH, DLC, DSNT, VDMT]):
    _IS_ELEM = True
    _ELEM_BY_CACHE = True
    FILESHANDLE_TYPE:type[FilesHandle] = DictLikeHandle

    SAVE_IMMIDIATELY = 0
    SAVE_AFTER_CLOSE = 1
    SAVE_STREAMLY = 2

    class StreamlyWriter(DisunifiedFileCluster._Writer):
        def __init__(self, cluster:"DictLikeCluster") -> None:
            super().__init__(cluster)
            self.streams:list[JsonIO.Stream] = []
            self.obj:DictLikeCluster = self.obj

        def enter_hook(self):
            self.obj.save_mode = self.obj.SAVE_STREAMLY
            streams = []
            for fh in self.obj.query_fileshandle(0, 'end'):
                fh: DictLikeHandle
                streams.append(JsonIO.Stream(fh.get_path(), True))
            self.streams.extend(streams)
            for stream in self.streams:
                stream.open()            
            super().enter_hook()


        def exit_hook(self):
            rlt = super().exit_hook()
            for stream in self.streams:
                stream.close()
            self.streams.clear()
            self.obj.save_mode = self.obj.SAVE_AFTER_CLOSE
            return rlt

        def write(self, data_i:int, elem_i, value):
            stream = self.streams[data_i]
            stream.write({elem_i: value})

    def __init__(self, dataset_node: Union[str, DatasetNode], name: str, *args, **kwargs) -> None:
        super().__init__(dataset_node, name, *args, **kwargs)
        self.__save_mode = self.SAVE_AFTER_CLOSE
        self.stream_writer = self.StreamlyWriter(self)
        
    @property
    def caches(self):
        return [fh.cache for fh in self.query_fileshandle(0, 'end')]

    @property
    def save_mode(self):
        return self.__save_mode
    
    @save_mode.setter
    def save_mode(self, mode):
        assert mode in [self.SAVE_IMMIDIATELY, self.SAVE_AFTER_CLOSE, self.SAVE_STREAMLY]
        if self.is_writing and mode != self.SAVE_STREAMLY:
            warnings.warn("can't change save_mode from SAVE_STREAMLY to the others while writing streamly")
        if self.__save_mode == self.SAVE_AFTER_CLOSE and mode != self.SAVE_AFTER_CLOSE and self.opened:
            self.save()
        self.__save_mode = mode

    @property
    def write_streamly(self):
        return self.save_mode == self.SAVE_STREAMLY

    def elem_keys(self):
        if len(self.MemoryData) == 0:
            return tuple()
        else:
            first_fh = next(iter(self.MemoryData.values()))
            return first_fh.cache.keys()

    ### IO ###
    def init_io_metas(self):
        super().init_io_metas()
        self.read_elem_meta = self._read_elem(self)
        self.write_elem_meta = self._write_elem(self)
        self.modify_elem_key_meta = self._modify_elem_key(self)
        self.remove_elem_meta = self._remove_elem(self)

    def __cvt_elem_i_input(self, elem_i:Union[int, Iterable[int]]):
        if isinstance(elem_i, int):
            return_list = False
            elem_i = [elem_i]
        elif isinstance(elem_i, Iterable):
            return_list = True
        else:
            raise TypeError(f"elem_i should be int or Iterable[int], not {type(elem_i)}")
        return return_list, elem_i

    class _read_elem(DisunifiedFileCluster._read["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, value, **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():    
                fh = self._query_fileshandle(data_i)
                rlt_dict[data_i] = fh.get_cache_at(src)
            return rlt_dict

    class _write_elem(DisunifiedFileCluster._write["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, values:dict[int, Any], **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():    
                fh = self._query_fileshandle(data_i)
                success = fh.set_cache_at(dst, values[data_i])
                if success and self.files_cluster.write_streamly:
                    self.files_cluster.stream_writer.write(data_i, dst, values[data_i])
                rlt_dict[data_i] = success
            return rlt_dict

    class _modify_elem_key(DisunifiedFileCluster._modify_key["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, values:dict, **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():
                elem = self._query_fileshandle(data_i).pop_cache_at(src)
                success = self._query_fileshandle(data_i).set_cache_at(dst, elem)
                rlt_dict[data_i] = success
            return rlt_dict

    class _remove_elem(DisunifiedFileCluster._remove["DictLikeCluster", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def operate_elem(self, src, dst, value, **other_paras):
            rlt_dict = {}
            for data_i in self.files_cluster.data_keys():
                rlt_dict[data_i] = self._query_fileshandle(data_i).pop_cache_at(dst)
            return rlt_dict        

    def read_elem(self, src:Union[int, Iterable[int]], *, force = False, **other_paras) -> Union[dict, list[dict]]:
        rlt = self.io_decorator(self.read_elem_meta, force=force)(src=src, **other_paras)
        return rlt

    def write_elem(self, dst:int, value:Union[Any, dict[int, Any]], *, force = False, **other_paras) -> dict:
        assert len(value) == self.data_num, f"values length {len(value)} != cluster length {self.data_num}"
        
        rlt = self.io_decorator(self.write_elem_meta, force=force)(dst=dst, value=value, **other_paras)

        if self.save_mode == self.SAVE_IMMIDIATELY:
            self.cache_to_file(force=True)
            self.save()
        return rlt

    def modify_elem_key(self, src:int, dst:int, *, force = False, **other_paras) -> dict:
        if self.write_streamly:
            raise ValueError("can't modify item while writing streamly")
        
        rlt = self.io_decorator(self.modify_elem_key_meta, force=force)(src=src, dst=dst, **other_paras)

        return rlt

    def remove_elem(self, dst:int, remove_both = False, *, force = False, **other_paras) -> dict:
        if self.write_streamly:
            raise ValueError("can't pop item while writing streamly")
        
        rlt = self.io_decorator(self.remove_elem_meta, force=force)(dst=dst, **other_paras)

        return rlt

    def sort_elem(self):
        if self.write_streamly:
            raise ValueError("can't pop item while writing streamly")
        for dict_fh in self.query_fileshandle(0, 'end'):
            dict_fh:DictLikeHandle
            dict_fh.sort_cache()
    ##########

    def _set_fileshandle(self, data_i, fh:DictLikeHandle):
        if (fh.elem_num == self.elem_num and fh.elem_continuous and self.elem_continuous) or (self.elem_num == 0 and self.data_num == 0):
            super()._set_fileshandle(data_i, fh)     
        else:       
            raise ValueError("can't add fileshandle while there are items or writing streamly")
        
    def add_default_file(self, filename):
        suffix = os.path.splitext(filename)[-1]
        assert suffix in FilesHandle.DEFAULT_FILE_TYPE, f"suffix {suffix} is not supported"
        assert issubclass(FilesHandle.DEFAULT_FILE_TYPE[suffix][2], dict), f"suffix {suffix} is not supported"
        fh = self.FILESHANDLE_TYPE.from_name(self, filename)
        self._set_fileshandle(self.data_i_upper, fh)

    def save_without_cache(self):
        '''
        to spare memory, save without cache
        '''
        dict_wo_cache = self.save_preprecess()
        for v in dict_wo_cache.values():
            
            v[FilesHandle.KW_cache][CacheProxy.KW_cache] = {}
        self.__class__.save_memory_func(self.MemoryData_path, dict_wo_cache)

    @classmethod
    def from_cluster(cls:type[DLC], cluster:DLC, dataset_node:DSNT = None, name = None, *args, **kwargs) -> DLC:
        new_cluster = super().from_cluster(cluster, dataset_node=dataset_node, name=name, *args, **kwargs)
        new_cluster.open()
        for fh in cluster.query_fileshandle(0, 'end'):
            new_fh = cls.FILESHANDLE_TYPE.from_fileshandle(new_cluster, fh, cache={})
            new_cluster._set_fileshandle(new_cluster.data_i_upper, new_fh)
        return new_cluster

    @staticmethod
    def Test():
        def print_all_fh(fc:DictLikeCluster):
            print()
            for fh in fc.query_fileshandle(0, 'end'):
                print(fh)

        top_dir = os.path.join(os.getcwd(), "DictLikeClusterTest")

        d0 = DictLikeCluster(top_dir, "000000")
        d1 = DictLikeCluster(top_dir, "000001")

        d0.clear(force=True)
        d1.clear(force=True)

        d0.add_default_file("f0.json")
        d0.add_default_file("f1.json")
        d0.add_default_file("f2.json")
        print_all_fh(d0)

        with d0.get_writer():
            d0.write(0, [np.array([1,2]), 2, None])
            d0.write(1, [np.array([3,4]), 4, -1])
        print_all_fh(d0)

        try:
            d0.add_default_file("f3.json")
        except:
            print("can not add fileshandle while there are items")
        
        d0.save_mode = d0.SAVE_IMMIDIATELY
        with d0.get_writer():
            d0.write(2, [np.array([5,6]), 6, -2])
        print_all_fh(d0)

        with d0.stream_writer:
            d0.write(2, [np.array([7,8]), 8, -3])
            d0.write(4, [np.array([9,10]), 10, -4])

        print("d0 elem_continuous:", d0.elem_continuous)
        d0.make_elem_continuous(True)
        print("d0 elem_continuous:", d0.elem_continuous)
        d0.save_without_cache()
        print_all_fh(d0)
        print(len(d0.elem_keys()))

        d1.copy_from(d0, cover=True, force=True)
        print_all_fh(d1)

        d1.file_to_cache(force=True)
        print_all_fh(d1)
        d1.cache_to_file(force=True)
        print_all_fh(d1)

        d1.rebuild()

        with d1.get_writer().allow_overwriting():
            d1.remove(0, remove_both=True)
        d1.make_continuous(True)

        print_all_fh(d1)

        d0.clear(force=True)
        d1.clear(force=True)      

class IntArrayDictCluster(UnifiedFileCluster[UFH, IADC, DSNT, dict[int, np.ndarray]], Generic[UFH, IADC, DSNT]):
    def __init__(self, dataset_node:DSNT, name: str, array_shape:tuple[int], array_fmt:str = "",
                 suffix:str = '.txt', *,
                 read_func:Callable[[str], VDMT] = None, 
                 write_func:Callable[[str, VDMT], None] = None, 
                 value_type:Callable = None,
                 filllen = 6, 
                 fillchar = '0',
                 alternate_suffix:list = None, 
                 **kwargs
                 ) -> None:
        read_func = partial(np.loadtxt, fmt = array_fmt) if read_func is None else read_func
        write_func = partial(np.savetxt, fmt = array_fmt) if write_func is None else write_func
        value_type = np.ndarray if value_type is None else value_type
        super().__init__(dataset_node, name, 
                         suffix=suffix, 
                         read_func=read_func, 
                         write_func=write_func, 
                         value_type=value_type, 
                         filllen=filllen, 
                         fillchar=fillchar, 
                         alternate_suffix=alternate_suffix, 
                         **kwargs)
        self.array_shape:tuple[int] = array_shape

    ### IO ####
    _FCT = TypeVar('_FCT', bound="IntArrayDictCluster")
    _VDMT = TypeVar('_VDMT', bound=dict[int, np.ndarray])
    _FHT = TypeVar('_FHT', bound=UnifiedFilesHandle)

    class _read(UnifiedFileCluster._read[_FCT, _VDMT, _FHT]):
        
        def inv_format_value(self, array:np.ndarray) -> "IntArrayDictCluster._VDMT":
            '''
            array: np.ndarray [N, 5]
            '''
            dict_ = {}
            for i in range(array.shape[0]):
                dict_[int(array[i, 0])] = array[i, 1:].reshape(self.files_cluster.array_shape)
            return dict_
    
    class _write(UnifiedFileCluster._write[_FCT, _VDMT, _FHT]):
        def format_value(self, value:"IntArrayDictCluster._VDMT") -> Any:
            array = []
            for i, (k, v) in enumerate(value.items()):
                array.append(
                    np.concatenate([np.array([k]).astype(v.dtype), v.reshape(-1)])
                    )
            array = np.stack(array)
            return array
 