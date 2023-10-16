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

from .IOAbstract import FilesCluster, FilesHandle, DatasetNode
from .datacluster import UnifiedFilesHandle, UnifiedFileCluster,\
                        DisunifiedFilesHandle, DisunifiedFileCluster,\
                        DictLikeCluster, DictLikeHandle
from .spliter import Spliter, SpliterGroup
# from . import Posture, JsonIO, JSONDecodeError, Table, extract_doc, search_in_dict, int_str_cocvt
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

FCT = TypeVar('FCT', bound='FilesCluster') # files cluster type
DST = TypeVar('DST', bound='Dataset') # dataset node type
VDST = TypeVar('VDST') # type of the value of dataset
from numpy import ndarray

def test_datasetnode():
    dsn = DatasetNode("./DatasetNodeTest", parent=None)
    dsn2 = DatasetNode("./DatasetNodeTest2", parent=None)

    dsn.add_cluster(UnifiedFileCluster(None, "ufc", 
                                       read_func=np.loadtxt, write_func=np.savetxt))
    dsn.add_cluster(DisunifiedFileCluster(None, "dfc"))
    dsn.add_cluster(DictLikeCluster(None, "dlc"))

    dsn.clear(force = True)

    dlc:DictLikeCluster = dsn.get_cluster("dlc")
    dlc.add_default_file("f0.json")

    ufc:UnifiedFileCluster = dsn.get_cluster("ufc")

    print(dsn.readonly)
    print(dlc.readonly)
    print(ufc.readonly)

    with dsn.get_writer():
        dsn.write(0, [np.array([1,2,3]), [0.5]])
        dsn.write(1, [np.array([1,2,3]), [0.5]])

    print(dsn.readonly)
    print(dlc.readonly)
    print(ufc.readonly)

    dsn.num

    dsn.read(0)

    dsn.remove(0, force= True)
    print(dsn.readonly)
    print(dlc.readonly)
    print(ufc.readonly)

    print(dsn.MemoryData)
    dsn.make_continuous(force=True)
    print(dsn.continuous)
    print(dlc.elem_continuous)
    print(ufc.elem_continuous)

    dsn.all_cache_to_file(force=True)
    print(dsn.MemoryData)


    dsn2.copy_from(dsn, force=True)
    print(dsn2.MemoryData)
    dsn2.clear(force=True)

def test_dataset():
    dsn = Dataset("./DatasetTest", parent=None)
    dsn2 = Dataset("./DatasetTest2", parent=None)

    # dsn.clear(force = True)

    dsn.add_cluster(UnifiedFileCluster(None, "ufc", 
                                       read_func=np.loadtxt, write_func=np.savetxt))
    dsn.add_cluster(DisunifiedFileCluster(None, "dfc"))
    dsn.add_cluster(DictLikeCluster(None, "dlc"))

    # dsn.reopen()

    dlc:DictLikeCluster = dsn.get_cluster("dlc")
    if "f0.json" not in dlc.file_names:
        dlc.add_default_file("f0.json")

    ufc:UnifiedFileCluster = dsn.get_cluster("ufc")
    dsn.active_spliter.is_writing
    with dsn.get_writer():
        for i in range(5):
            dsn.append([np.array([1,2,3]) * i, [np.array([i])]])

    dsn.active_spliter.set_all_by_rate([0.6, 0.4])
    print(dsn.active_spliter.get_nums())
    print(dsn.MemoryData)
    print(dsn.active_spliter.split_table)
    dsn.num
    dsn.read(0)

    dsn.remove(0, force= True)

    print(dsn.MemoryData)
    print(dsn.spliter_group.MemoryData)
    dsn.make_continuous(force=True)
    print(dsn.continuous)
    print(dlc.continuous)
    print(ufc.continuous)

    dsn.all_cache_to_file(force=True)
    print(dsn.MemoryData)

    dsn2.copy_from(dsn, force=True)
    print(dsn2.MemoryData)
    dsn2.all_cache_to_file(force=True)
    dsn2.clear(force=True, clear_completely=True)


class Dataset(DatasetNode[FCT, DST, VDST], Generic[FCT, DST, VDST]):
    def init_clusters_hook(self):
        super().init_clusters_hook()
        self.spliter_group = SpliterGroup("split_group", parent=self)

    @property
    def split_mode(self):
        return self.spliter_group.split_mode
    
    @split_mode.setter
    def split_mode(self, value):
        self.spliter_group.split_mode = value
    
    @property
    def active_spliter(self):
        return self.spliter_group.active_spliter

    @property
    def train_idx_array(self):
        return self.spliter_group.train_idx_list
    
    @property
    def val_idx_array(self):
        return self.spliter_group.val_idx_list
    
    @property
    def default_train_idx_array(self):
        return self.spliter_group.get_cluster(SpliterGroup.DEFAULT_SPLIT_MODE[0]).get_idx_list(Spliter.KW_TRAIN)
    
    @property
    def default_val_idx_array(self):
        return self.spliter_group.get_cluster(SpliterGroup.DEFAULT_SPLIT_MODE[0]).get_idx_list(Spliter.KW_VAL)
    
    def update_overview(self, log_type, src, dst, value, cluster:FilesCluster):
        super().update_overview(log_type, src, dst, value, cluster)
        if log_type == self.LOG_READ or\
           log_type == self.LOG_CHANGE or\
           log_type == self.LOG_OPERATION:
            return
        if log_type == self.LOG_ADD:
            self.spliter_group.add_elem(dst)
        if log_type == self.LOG_REMOVE:
            if dst not in self.keys():
                self.spliter_group.remove_elem(dst)
        if log_type == self.LOG_MOVE:
            if src in self.keys():
                self.spliter_group.copy_elem(src, dst)
            if src not in self.keys():
                self.spliter_group.remove_elem(src)