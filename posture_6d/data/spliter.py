# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_items, dict_keys, dict_values
from collections.abc import Iterator
from MyLib.posture_6d.data.IOAbstract import FilesCluster
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
from .IOAbstract import DataMapping, DatasetNode, IOMeta, BinDict, _KT, _VT, DMT, VDMT, DSNT,\
    FilesHandle, CacheProxy, \
    AmbiguousError, IOMetaParameterError, KeyNotFoundError, ClusterDataIOError, DataMapExistError, \
        IOStatusWarning, ClusterIONotExecutedWarning, ClusterNotRecommendWarning,\
    FilesCluster,\
    parse_kw
from .datacluster import DisunifiedFilesHandle, DisunifiedFileCluster, DictLikeCluster, DictLikeHandle

SFH = TypeVar('SFH', bound='SpliterFilesHandle')
SP = TypeVar('SP', bound='Spliter')
SPG = TypeVar('SPG', bound='SpliterGroup')
VDMT = TypeVar('VDMT') # type of the value of data cluster


'''
spliter(DisunifiedFilesHandle) .json  维护一个table, 行：elem_i, 列：sub_set, 值：bool
行、列增删，元素读取；设置idx所在的subset；是否独占

splitergroup(Cluster) 维护多个spliter, 每个spliter是一个 split_mode
'''

class SpliterFilesHandle(DisunifiedFilesHandle[SP, Table[int, str, bool]]):
    DEFAULT_READ_FUNC = Table.from_json
    DEFAULT_WRITE_FUNC = Table.to_json
    DEFAULT_VALUE_TYPE = Table
    DEFAULT_VALUE_INIT_FUNC = partial(Table, row_name_type=int, col_name_type=str, default_value_type=bool)

    LOAD_CACHE_ON_INIT = True

# class Spliter(DisunifiedFileCluster[SpliterFilesHandle, SP, SPG, Table[int, str, bool]], Generic[SP, SPG]):
#     SPLIT_FILE = "split.json"
#     FILESHANDLE_TYPE = SpliterFilesHandle

#     KW_TRAIN = "train"
#     KW_VAL = "val"
#     DEFAULT_SUB_SET = [KW_TRAIN, KW_VAL]

#     def __init__(self, dataset_node: Union[str, DatasetNode], name: str, *args, **kwargs) -> None:
#         super().__init__(dataset_node, name, *args, **kwargs)
#         self.split_fileshandle:SpliterFilesHandle = self.FILESHANDLE_TYPE.from_name(self, self.SPLIT_FILE)
#         self._set_fileshandle(0, self.split_fileshandle)

#         self.__exclusive = False

#         with self.get_writer():
#             for subset in self.DEFAULT_SUB_SET:
#                 self.add_subset(subset)

#     @property
#     def exclusive(self):
#         return self.__exclusive
    
#     @exclusive.setter
#     def exclusive(self, value):
#         self.__exclusive = bool(value)

#     @property
#     def split_table(self):
#         return self.split_fileshandle.cache
    
#     @property
#     def split_table_data(self):
#         return self.split_table.data
    
#     @property
#     def subsets(self):
#         return self.split_table.col_names

#     #### add\remove\set\query of elem/subset ####
#     def add_elem(self, elem_i:int):
#         self.split_table.add_row(elem_i, exist_ok=True)
    
#     def remove_elem(self, elem_i:int):
#         self.split_table.remove_row(elem_i, not_exist_ok=True)

#     def add_subset(self, subset:str):
#         self.split_table.add_column(subset, exist_ok=True)

#     def remove_subset(self, subset:str):
#         self.split_table.remove_column(subset, not_exist_ok=True)

#     def set_one(self, elem_i:int, subset:str, value:bool):
#         if elem_i not in self.split_table.row_names:
#             self.add_elem(elem_i)
#         if self.exclusive and value:
#             self.split_table[elem_i, :] = False
#         self.split_table[elem_i, subset] = value
    
#     def set_one_elem(self, elem_i, subsets:dict[str, bool]):
#         for subset, value in subsets.items():
#             if subset not in self.split_table.col_names:
#                 raise KeyError(f"subset {subset} not in {self.split_table.col_names}")
#             self.set_one(elem_i, subset, value)
    
#     def set_one_subset(self, subset:str, elems:dict[int, bool]):
#         for elem_i, value in elems.items():
#             self.set_one(elem_i, subset, value)
    
#     def qurey_one(self, elem_i:int, subset:str):
#         return self.split_table[elem_i, subset]
    
#     def qurey_one_elem(self, elem_i:int):
#         return self.split_table.get_row(elem_i)
    
#     def qurey_one_subset(self, subset:str):
#         return self.split_table.get_column(subset)
    
#     def set_one_by_rate(self, elem_i, split_rate):
#         split_rate = self.process_split_rate(split_rate, len(self.subsets))
#         total_nums = [sum(self.split_table.get_column(sub).values()) for sub in self.subsets]
#         if sum(total_nums) == 0:
#             # all empty, choose the first
#             subset_idx = 0
#         else:
#             rates = np.array(total_nums) / sum(total_nums)
#             subset_idx = 0
#             for idx, r in enumerate(rates):
#                 if r <= split_rate[idx]:
#                     subset_idx = idx
#                     break
#         self.set_one(elem_i, self.subsets[subset_idx], True)
#         return self.subsets[subset_idx]
    
#     def set_all_by_rate(self, split_rate):
#         split_rate = self.process_split_rate(split_rate, len(self.subsets))
#         for elem_i in self.split_table.row_names:
#             self.set_one_by_rate(elem_i, split_rate)

#     @staticmethod
#     def process_split_rate(split_rate:Union[float, Iterable[float], dict[str, float]], 
#                             split_num:Union[int, list[str]]):
#         if isinstance(split_num, int):
#             subsets = None
#         elif isinstance(split_num, Iterable):
#             subsets = split_num
#             split_num = len(subsets)
#         else:
#             raise ValueError("split_num must be int or Iterable")
        
#         assert split_num > 1, "len(subsets) must > 1"
        
#         if split_num == 2 and isinstance(split_rate, float):
#             split_rate = (split_rate, 1 - split_rate)
#         if subsets is not None and isinstance(split_rate, dict):
#             split_rate = tuple([split_rate[subset] for subset in subsets])
#         elif isinstance(split_rate, Iterable):
#             split_rate = tuple(split_rate)
#         else:
#             raise ValueError("split_rate must be Iterable or dict[str, float], (or float if len(subsets) == 2)")
#         assert len(split_rate) == split_num, "splite_rate must have {} elements".format(split_num)
        
#         return split_rate

#     ##### as idx list #####
#     def get_nums(self):
#         return {subset: sum(self.split_table.get_column(subset).values()) for subset in self.subsets}

#     def get_idx_list(self, subset:Union[str, int]) -> list[int]:
#         ok_dict = self.split_table.get_column(subset)
#         return tuple([elem_i for elem_i, ok in ok_dict.items() if ok])
    
#     def save_as_txt(self, mask_mode = True):
#         save_paths = [os.path.join(self.data_path, f"{name}.txt") for name in self.subsets]
#         save_paths_dict = {subset: save_path for subset, save_path in zip(self.subsets, save_paths)} # subset: save_path
#         save_array = {}
#         if mask_mode:
#             all_elem_i = self.split_table.row_names
#             for subset, save_path in save_paths_dict.items():
#                 array = np.full((len(all_elem_i), 2), -1, dtype=int)
#                 idx_list = self.get_idx_list(subset)
#                 array[np.isin(array, idx_list), -1] = np.array(idx_list)
#                 save_array[subset] = array
#         else:
#             for subset, save_path in save_paths_dict.items():
#                 idx_list = self.get_idx_list(subset)
#                 save_array[subset] = np.array(idx_list)
        
#         for subset, array in save_array.items():
#             np.savetxt(save_paths_dict[subset], array, fmt='%d')

#     @classmethod
#     def from_txt(cls, txt_name_list:list[str]):
#         raise NotImplementedError
    
#     def __str__(self) -> str:
#         return f"{self.identity_string()} :: {str(self.get_nums())}"

class Spliter(DisunifiedFileCluster[SpliterFilesHandle, SP, SPG, Table[int, str, bool]], Generic[SP, SPG]):
    SPLIT_FILE = "split.json"
    FILESHANDLE_TYPE = SpliterFilesHandle

    KW_TRAIN = "train"
    KW_VAL = "val"
    DEFAULT_SUB_SET = [KW_TRAIN, KW_VAL]

    _IS_ELEM = True
    _ELEM_BY_CACHE = True

    ALWAYS_ALLOW_WRITE = True

    def __init__(self, dataset_node: Union[str, DatasetNode], name: str, *args, **kwargs) -> None:
        super().__init__(dataset_node, name, *args, **kwargs)
        self.split_fileshandle:SpliterFilesHandle = self.FILESHANDLE_TYPE.from_name(self, self.SPLIT_FILE)
        self._set_fileshandle(0, self.split_fileshandle)
        
        self.__exclusive = False

        for subset in self.DEFAULT_SUB_SET:
            self.add_subset(subset)

    @property
    def exclusive(self):
        return self.__exclusive
    
    @exclusive.setter
    def exclusive(self, value):
        self.__exclusive = bool(value)

    @property
    def split_table(self):
        return self.split_fileshandle._unsafe_get_cache()
    
    @property
    def split_table_data(self):
        return self.split_table.data
    
    @property
    def elem_idx(self):
        return self.split_table.row_names

    @property
    def subsets(self):
        return self.split_table.col_names

    #### add\remove\set\query of elem/subset ####

    def elem_keys(self):
        return self.split_table.keys()
    
    ### IO ###
    def init_io_metas(self):
        super().init_io_metas()
        self.read_elem_meta = self._read_elem(self)
        self.write_elem_meta = self._write_elem(self)
        self.modify_elem_key_meta = self._modify_elem_key(self)
        self.remove_elem_meta = self._remove_elem(self)

    class _read_elem(DisunifiedFileCluster._read["Spliter", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        @property
        def split_table(self) -> Table[int, str, bool]:
            return self.files_cluster.split_table

        def operate_elem(self, src, dst, value, **other_paras):
            return self.split_table.get_row(src)

    class _write_elem(DisunifiedFileCluster._write["Spliter", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        @property
        def split_table(self) -> Table[int, str, bool]:
            return self.files_cluster.split_table
        
        def is_overwriting(self, dst: int):
            return self.files_cluster.one_defined(dst)

        def operate_elem(self, src, dst, values:dict[str, Any], **other_paras):
            self.split_table.add_row(dst, exist_ok=True)
            for subset, value in values.items():
                if subset not in self.split_table.col_names:
                    raise KeyError(f"subset {subset} not in {self.split_table.col_names}")
                if self.files_cluster.exclusive and value:
                    self.split_table[dst, :] = False
                self.split_table[dst, subset] = value
            return None

    class _modify_elem_key(DisunifiedFileCluster._modify_key["Spliter", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def is_overwriting(self, dst: int):
            return self.files_cluster.one_defined(dst)
        @property
        def split_table(self) -> Table[int, str, bool]:
            return self.files_cluster.split_table
        def operate_elem(self, src, dst, values:dict, **other_paras):
            self.split_table.move_row(src, dst)

    class _remove_elem(DisunifiedFileCluster._remove["Spliter", dict[int, Any], DictLikeHandle]):
        OPER_ELEM = True
        def is_overwriting(self, dst: int):
            return self.files_cluster.one_defined(dst)

        def operate_elem(self, src, dst, value, **other_paras):
            self.files_cluster.split_table.remove_row(dst, not_exist_ok=True)

    def read_elem(self, src:Union[int, Iterable[int]], *, force = False, **other_paras) -> Union[dict[str, bool], None]:
        rlt = self.io_decorator(self.read_elem_meta, force=force)(src=src, **other_paras)
        return rlt

    def write_elem(self, dst:int, value:Union[Any, dict[int, Any]], *, force = False, **other_paras) -> dict:
        rlt = self.io_decorator(self.write_elem_meta, force=force)(dst=dst, value=value, **other_paras)
        return rlt

    def modify_elem_key(self, src:int, dst:int, *, force = False, **other_paras) -> dict:
        rlt = self.io_decorator(self.modify_elem_key_meta, force=force)(src=src, dst=dst, **other_paras)
        return rlt

    def remove_elem(self, dst:int, remove_both = False, *, force = False, **other_paras) -> dict:
        rlt = self.io_decorator(self.remove_elem_meta, force=force)(dst=dst, **other_paras)
        return rlt
    
    def sort_elem(self):
        self.split_table.sort()
    ##########

    def _set_fileshandle(self, data_i, fh:DictLikeHandle):
        if self.data_num > 0:
            raise ValueError("Spliter can only have one fileshandle")
        else:
            super()._set_fileshandle(data_i, fh)

    def add_elem_idx(self, elem_i:int):
        self.write_elem(elem_i, {})

    def remove_elem_idx(self, elem_i:int):
        self.remove_elem(elem_i)

    def add_subset(self, subset:str):
        self.split_table.add_column(subset, exist_ok=True)

    def remove_subset(self, subset:str):
        self.split_table.remove_column(subset, not_exist_ok=True)

    def set_one(self, elem_i:int, subset:str, value:bool):
        self.write_elem(elem_i, {subset: value})
    
    def set_one_elem(self, elem_i, subsets:dict[str, bool]):
        self.write_elem(elem_i, subsets)
    
    def set_one_subset(self, subset:str, elems:dict[int, bool]):
        for elem_i, value in elems.items():
            self.set_one(elem_i, subset, value)
    
    def one_defined(self, elem_i:int):
        if elem_i not in self.elem_keys():
            return False
        else:
            elem = self.qurey_one_elem(elem_i)
            if elem is None:
                return False
            else:
                return any(elem.values())

    def qurey_one(self, elem_i:int, subset:str):
        return self.read_elem(elem_i)[subset]
    
    def qurey_one_elem(self, elem_i:int):
        return self.read_elem(elem_i)
    
    def qurey_one_subset(self, subset:str):
        return self.split_table.get_column(subset)
    
    def set_one_by_rate(self, elem_i, split_rate):
        split_rate = self.process_split_rate(split_rate, len(self.subsets))
        total_nums = [sum(self.split_table.get_column(sub).values()) for sub in self.subsets]
        if sum(total_nums) == 0:
            # all empty, choose the first
            subset_idx = 0
        else:
            rates = np.array(total_nums) / sum(total_nums)
            subset_idx = 0
            for idx, r in enumerate(rates):
                if r <= split_rate[idx]:
                    subset_idx = idx
                    break
        self.set_one(elem_i, self.subsets[subset_idx], True)
        return self.subsets[subset_idx]
    
    def set_all_by_rate(self, split_rate):
        split_rate = self.process_split_rate(split_rate, len(self.subsets))
        with self.get_writer().allow_overwriting(True):
            for elem_i in self.split_table.row_names:
                self.set_one_by_rate(elem_i, split_rate)

    @staticmethod
    def process_split_rate(split_rate:Union[float, Iterable[float], dict[str, float]], 
                            split_num:Union[int, list[str]]):
        if isinstance(split_num, int):
            subsets = None
        elif isinstance(split_num, Iterable):
            subsets = split_num
            split_num = len(subsets)
        else:
            raise ValueError("split_num must be int or Iterable")
        
        assert split_num > 1, "len(subsets) must > 1"
        
        if split_num == 2 and isinstance(split_rate, float):
            split_rate = (split_rate, 1 - split_rate)
        if subsets is not None and isinstance(split_rate, dict):
            split_rate = tuple([split_rate[subset] for subset in subsets])
        elif isinstance(split_rate, Iterable):
            split_rate = tuple(split_rate)
        else:
            raise ValueError("split_rate must be Iterable or dict[str, float], (or float if len(subsets) == 2)")
        assert len(split_rate) == split_num, "splite_rate must have {} elements".format(split_num)
        
        return split_rate

    ##### as idx list #####
    def get_nums(self):
        return {subset: sum(self.split_table.get_column(subset).values()) for subset in self.subsets}

    def get_idx_list(self, subset:Union[str, int]) -> list[int]:
        ok_dict = self.split_table.get_column(subset)
        return tuple([elem_i for elem_i, ok in ok_dict.items() if ok])
    
    def save_as_txt(self, mask_mode = True):
        save_paths = [os.path.join(self.data_path, f"{name}.txt") for name in self.subsets]
        save_paths_dict = {subset: save_path for subset, save_path in zip(self.subsets, save_paths)} # subset: save_path
        save_array = {}
        if mask_mode:
            all_elem_i = self.split_table.row_names
            for subset, save_path in save_paths_dict.items():
                array = np.full((len(all_elem_i), 2), -1, dtype=int)
                idx_list = self.get_idx_list(subset)
                array[np.isin(array, idx_list), -1] = np.array(idx_list)
                save_array[subset] = array
        else:
            for subset, save_path in save_paths_dict.items():
                idx_list = self.get_idx_list(subset)
                save_array[subset] = np.array(idx_list)
        
        for subset, array in save_array.items():
            np.savetxt(save_paths_dict[subset], array, fmt='%d')

    @classmethod
    def from_txt(cls, txt_name_list:list[str]):
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.identity_string()} :: {str(self.get_nums())}"


class SpliterGroup(DatasetNode[Spliter, SPG, Table[int, str, bool]], Generic[SP, SPG]):
    DEFAULT_SPLIT_MODE = ["default"]

    def init_clusters_hook(self):
        super().init_clusters_hook()
        for split_mode in self.DEFAULT_SPLIT_MODE:
            self.add_cluster(Spliter(self, split_mode))
    
    def init_dataset_attr(self):
        super().init_dataset_attr()

        self.__split_mode = self.DEFAULT_SPLIT_MODE[0]

    @property
    def split_mode(self):
        return self.__split_mode
    
    @split_mode.setter
    def split_mode(self, value):
        if value not in self.clusters_map:
            raise KeyError(f"split_mode {value} not in {self.cluster_keys()}")
        self.__split_mode = value

    @property
    def active_spliter(self):
        return self.clusters_map[self.__split_mode]
    
    @property
    def train_idx_list(self):
        return self.active_spliter.get_idx_list("train")
    
    @property
    def val_idx_list(self):
        return self.active_spliter.get_idx_list("val")
    
    def add_elem(self, elem_i:int):
        for spliter in self.clusters_map.values():
            spliter.add_elem_idx(elem_i)

    def remove_elem(self, elem_i:int):
        for spliter in self.clusters_map.values():
            spliter.remove_elem_idx(elem_i)

    def move_elem(self, src:int, dst:int):
        for spliter in self.clusters_map.values():
            spliter.modify_elem_key(src, dst)
    
    def copy_elem(self, src:int, dst:int):
        for spliter in self.clusters_map.values():
            spliter.write_elem(dst, spliter.read_elem(src))