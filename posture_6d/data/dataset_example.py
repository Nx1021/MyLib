from typing import Callable

from .dataset import Dataset, DatasetNode, Mix_Dataset
from .dataCluster import UnifiedFileCluster, DisunifiedFileCluster, DictLikeCluster, UnifiedFilesHandle, DisunifiedFilesHandle, DictLikeHandle,\
    IntArrayDictAsTxtCluster, NdarrayAsTxtCluster
from .IOAbstract import ClusterWarning, FilesCluster
from .viewmeta import ViewMeta
from ..core.utils import deserialize_object, serialize_object, rebind_methods

import numpy as np
import cv2
import warnings
from functools import partial


class BopFormat(Dataset):
    pass

class cxcywhLabelCluster(IntArrayDictAsTxtCluster[UnifiedFilesHandle, "cxcywhLabelCluster", "VocFormat_6dPosture"]):
    KW_IO_RAW = "raw"

    def init_attrs(self):
        super().init_attrs()
        self.default_image_size = None
    
    class _read(IntArrayDictAsTxtCluster._read["cxcywhLabelCluster", dict[int, np.ndarray], UnifiedFilesHandle]):
        def postprogress_value(self, value:np.ndarray, *, image_size = None, **other_paras):
            value = super().postprogress_value(value)
            image_size = self.files_cluster.default_image_size if image_size is None else image_size
            if image_size is not None:
                bbox_2d = value[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = cxcywhLabelCluster._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
                value[:,1:] = bbox_2d   
            else:
                if image_size != self.files_cluster.KW_IO_RAW:
                    warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2",
                    ClusterWarning)
            return value
        
    class _write(IntArrayDictAsTxtCluster._write["cxcywhLabelCluster", dict[int, np.ndarray], UnifiedFilesHandle]):
        def preprogress_value(self, value:dict[int, np.ndarray], *, image_size = None, **other_paras):
            value = super().preprogress_value(value)
            image_size = self.files_cluster.default_image_size if image_size is None else image_size
            if image_size is not None:
                bbox_2d = {}
                for k, v in value.items():
                    bbox_2d[k] = cxcywhLabelCluster._x1y1x2y2_2_normedcxcywh(v, image_size)
                value = bbox_2d
            else:
                if image_size != self.files_cluster.KW_IO_RAW:
                    warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh",
                    ClusterWarning)
            return value

    def set_rely(self, relied: FilesCluster, rlt):
        if relied.flag_name == ViewMeta.COLOR:
            rlt:np.ndarray = rlt
            self._relied_paras.update({"image_size": rlt.shape[:2][::-1]})

    def read(self, src: int, *, sub_dir = None, image_size = None, force = False,  **other_paras) -> dict[int, np.ndarray]:
        return super().read(src, sub_dir=sub_dir, image_size = image_size, force=force, **other_paras)
    
    def write(self, data_i: int, value: dict[int, np.ndarray], *, sub_dir = None, image_size = None, force = False, **other_paras):
        return super().write(data_i, value, sub_dir=sub_dir, image_size = image_size, force=force, **other_paras)

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

class VocFormat_6dPosture(Mix_Dataset[UnifiedFileCluster, "VocFormat_6dPosture", ViewMeta]):
    KW_IMGAE_DIR = "images"

    POSTURE_SPLITER_NAME = "posture"
    POSTURE_SUBSETS = ["train", "val"]


    SPLIT_PARA = Mix_Dataset.SPLIT_PARA.copy()
    SPLIT_PARA.update(
        {
            POSTURE_SPLITER_NAME: POSTURE_SUBSETS,
        }
    )

    def __init__(self, directory, *, flag_name="", parent: DatasetNode = None) -> None:
        super().__init__(directory, flag_name=flag_name, parent=parent)

    def init_clusters_hook(self):
        super().init_clusters_hook()
        self.images_elements =\
              UnifiedFileCluster[UnifiedFilesHandle, UnifiedFileCluster, VocFormat_6dPosture, np.ndarray](self, self.KW_IMGAE_DIR, suffix = ".jpg" ,
                    read_func=cv2.imread, 
                    write_func=cv2.imwrite, 
                    flag_name=ViewMeta.COLOR)
        
        self.depth_elements      =\
              UnifiedFileCluster[UnifiedFilesHandle, UnifiedFileCluster, VocFormat_6dPosture, np.ndarray](self, "depths",  suffix = '.png',
                    read_func = partial(cv2.imread, flags = cv2.IMREAD_ANYDEPTH), 
                    write_func =cv2.imwrite,
                    flag_name=ViewMeta.DEPTH)
        
        self.masks_elements      =\
              UnifiedFileCluster[UnifiedFilesHandle, UnifiedFileCluster, VocFormat_6dPosture, dict[int, np.ndarray]](self, "masks", suffix = ".pkl",
                    read_func = deserialize_object,
                    write_func = serialize_object,
                    flag_name=ViewMeta.MASKS)
        rebind_methods(self.masks_elements.read_meta, self.masks_elements.read_meta.inv_format_value, self.deserialize_mask_dict) ### TODO
        rebind_methods(self.masks_elements.write_meta, self.masks_elements.write_meta.format_value, self.serialize_mask_dict) ### TODO

        self.extr_vecs_elements  = IntArrayDictAsTxtCluster(self, "trans_vecs",     array_shape=(2, 3), write_func_kwargs={"fmt":"%8.8f"},  
                                                            flag_name=ViewMeta.EXTR_VECS)

        self.intr_elements       = NdarrayAsTxtCluster(self,        "intr",         array_shape=(3,3),  write_func_kwargs={"fmt":"%8.8f", "delimiter":'\t'},
                                                            flag_name=ViewMeta.INTR)

        self.depth_scale_elements= NdarrayAsTxtCluster(self,        "depth_scale",  array_shape=(-1,),  write_func_kwargs={"fmt":"%8.8f"}, 
                                                            flag_name=ViewMeta.DEPTH_SCALE)

        self.bbox_3ds_elements   = IntArrayDictAsTxtCluster(self,   "bbox_3ds",     array_shape=(-1, 2), write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.BBOX_3DS)

        self.landmarks_elements  = IntArrayDictAsTxtCluster(self,   "landmarks",    array_shape=(-1, 2), write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.LANDMARKS)

        self.visib_fracts_element = IntArrayDictAsTxtCluster(self,   "visib_fracts", array_shape=(-1,),  write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.VISIB_FRACTS)

        self.labels_elements      = cxcywhLabelCluster(self,         "labels",       array_shape=(-1,),  write_func_kwargs={"fmt":"%8.8f"},
                                                            flag_name=ViewMeta.LABELS)

        self.labels_elements.default_image_size = (640, 480)
        self.labels_elements.link_rely_on(self.images_elements)

    @staticmethod
    def serialize_mask_dict(obj, mask_ndarray_dict:dict[int, np.ndarray]):
        def serialize_image(image:np.ndarray):  
            # 将NumPy数组编码为png格式的图像
            retval, buffer = cv2.imencode('.png', image)
            # 将图像数据转换为字节字符串
            image_bytes = buffer.tobytes()
            image.tobytes()
            return image_bytes

        new_value = dict(zip(mask_ndarray_dict.keys(), [serialize_image(x) for x in mask_ndarray_dict.values()]))
        return new_value

    @staticmethod
    def deserialize_mask_dict(obj, mask_bytes_dict:dict[int, bytes]):
        def deserialize_image(image_bytes):  
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array)# 将numpy数组解码为图像
            return image
        new_value = dict(zip(mask_bytes_dict.keys(), [deserialize_image(x) for x in mask_bytes_dict.values()]))
        return new_value

    @property
    def default_spliter(self):
        return self.spliter_group.get_cluster("default")

    def init_dataset_attr_hook(self):
        super().init_dataset_attr_hook()
        self.default_split_rate = 0.75

    def get_default_set_of(self, data_i):
        default_split_dict = self.default_spliter.get_idx_dict()
        for sub_set, idx_array in default_split_dict.items():
            if data_i in idx_array:
                return sub_set
        # didn't find data_i in default_split_dict
        sub_set = self.default_spliter.set_one_by_rate(data_i, self.default_split_rate)
        return sub_set
    
    def read(self, src: int, *, force = False, **other_paras)-> ViewMeta:
        raw_read_rlt = super().raw_read(src, force = force, **other_paras)
        return ViewMeta(**raw_read_rlt)
    
    def write(self, data_i: int, value: ViewMeta, *, force = False, **other_paras):
        sub_dir = self.get_default_set_of(data_i)
        super().raw_write(data_i, value.as_dict(), sub_dir = sub_dir, force = force, **other_paras)
