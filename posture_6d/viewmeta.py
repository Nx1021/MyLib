# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import cv2
from PIL import Image
import io
import copy
import pickle
from typing import Union, Any, Callable
from .posture import Posture
from .intr import CameraIntr
from .mesh_manager import MeshMeta
from .derive import calc_bbox_3d_proj, calc_landmarks_proj, calc_masks
from .utils import get_bbox_connections, modify_class_id, get_meta_dict
import inspect
import warnings

def query_key_by_value(orig_dict:dict):
    __orig_value_ids    = [id(x) for x in orig_dict.values()]
    __orig_keys         = list(orig_dict.keys())
    def query(value):
        matched_idx = __orig_value_ids.index(id(value))
        key = __orig_keys[matched_idx]
        __orig_value_ids.pop(matched_idx)
        __orig_keys.pop(matched_idx)
        return matched_idx, key
    return query

def copy_by_rect(crop_rect, org:np.ndarray):
    if len(crop_rect.shape) == 1:
        crop_rect = np.expand_dims(crop_rect, 0)
    assert len(crop_rect.shape) == 2 and crop_rect.shape[1] == 4, "the shape of crop_rect must be [N, 4]"
    if not np.issubdtype(crop_rect.dtype, np.integer):
        crop_rect = np.round(crop_rect).astype(np.int32)
    new = np.zeros(org.shape, org.dtype)
    for r in crop_rect:
        new[r[1]: r[3], r[0]: r[2]] = \
            org[r[1]: r[3], r[0]: r[2]]
    return new

def rotate_image(M, image:np.ndarray):
    # 执行旋转
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def rot_xy_list_2dpoints(M, points_2d:np.ndarray) -> list[float]:
    '''
    points_2d [N, (x,y)]
    '''
    points_2d = np.pad(points_2d, ((0,0),(0,1)), constant_values=1)
    new_points_2d:np.ndarray = M.dot(points_2d.T).T[:, :2]
    return new_points_2d

def ignore_viewmeta_warning(func):
    def wrapper(*arg, **kw):
        ViewMeta.IGNORE_WARNING = True
        result = func(*arg, *kw)
        ViewMeta.IGNORE_WARNING = False
        return result
    return wrapper

def calc_bbox2d_from_mask_dict(mask_dict:dict[int, np.ndarray]):
    bbox_2d = {}
    for id_, mask in mask_dict.items():
        where = np.where(mask)
        if where[0].size == 0:
            bbox_2d[id_] = np.array([0, 0, 0, 0]).astype(np.int32)
        else:
            lt = np.min(where, -1)
            rb = np.max(where, -1)
            bbox_2d[id_] = np.array([lt[1], lt[0], rb[1], rb[0]])
    return bbox_2d

class ViewMeta():
    '''
    一个视角下的所有数据的元
    '''
    class AugmentPipeline():
        def get_ap_of_meta(self, class_)-> Union["ViewMeta.AugmentPipeline", None]:
            for value in self.meta.agmts.values():
                if isinstance(value, class_):
                    return value
            return None

        def __init__(self, meta:"ViewMeta") -> None:
            self.meta = meta
            self.new_obj = None

        @property
        def obj(self):
            return None

        def __inner_func(self, func, *args):
            if self.obj is None:
                return None
            else:
                self.new_obj = func(*args)
                return self.new_obj

        def _crop(self, crop_rect: ndarray):
            return self.obj
        
        def crop(self, crop_rect: ndarray)-> Union[cv2.Mat, None]:
            '''
            crop_rect: [N, [y1, x1, y2, x2]] int
            '''
            return self.__inner_func(self._crop, crop_rect)

        def _rotate(self, M: ndarray):
            return self.obj

        def rotate(self, M: ndarray):
            return self.__inner_func(self._rotate, M)

        def _change_brightness(self, delta_value, direction):
            return self.obj

        def change_brightness(self, delta_value:float, direction:tuple[float]=(0, 0)):
            return self.__inner_func(self._change_brightness, delta_value, direction)

        def _change_saturation(self, delta_value):
            return self.obj

        def change_saturation(self, delta_value:float):
            return self.__inner_func(self._change_saturation, delta_value)

    class ColorAP(AugmentPipeline):
        '''
        Color image augment pipline
        '''
        def __init__(self, meta:"ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.color

        def _crop(self, crop_rect: np.ndarray):
            return copy_by_rect(crop_rect, self.obj)

        def _rotate(self, M: ndarray):
            return rotate_image(M, self.obj)
        
        def _change_brightness(self, delta_value, direction):
            direction = np.array(direction, np.float32)
            # 创建一个640x480的网格
            x, y = np.meshgrid(np.arange(self.obj.shape[0]), np.arange(self.obj.shape[1]))
            # 将网格转换为 (640*480, 2) 的坐标数组
            coords = np.stack((x.ravel(), y.ravel()), axis=1)
            value = np.sum(coords * direction, axis=1)
            if value.max() == 0:
                value[:] = 1 # 亮度整体变化
            else:
                value -= value.min()
                value = (value / value.max() - 0.5) * delta_value #沿着梯度方向变换

            hsv = cv2.cvtColor(self.obj, cv2.COLOR_BGR2HSV)
            v = hsv[:,:,2].astype(np.float32)
            v[tuple(coords.T)] += value
            v = np.clip(v, 0, 255)
            hsv[:,:,2] = v.astype(np.uint8)
            new_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return new_color
        
        def _change_saturation(self, delta_value):
            hsv = cv2.cvtColor(self.obj, cv2.COLOR_BGR2HSV)
            s = hsv[:,:,1].astype(np.float32)
            s += delta_value
            s = np.clip(s, 0, 255)
            hsv[:,:,1] = s.astype(np.uint8)
            new_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return new_color

    class DepthAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.depth

        def _crop(self, crop_rect: np.ndarray):
            new_depth = copy_by_rect(crop_rect, self.obj)
            return new_depth
        
        def _rotate(self, M: ndarray):
            new_depth = rotate_image(M, self.obj)
            return new_depth

    class MasksAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.masks

        def _crop(self, crop_rect: np.ndarray):
            new_masks:dict[str, cv2.Mat] = {}
            for _id, mask in self.obj.items():
                new_mask = copy_by_rect(crop_rect, mask)
                new_masks.update({_id: new_mask})
            return new_masks
        
        def _rotate(self, M:ndarray):
            new_masks = {}
            # 裁剪所有mask
            for _id, mask in self.obj.items():
                if np.issubdtype(mask.dtype, np.bool_):
                    mask = mask.astype(np.uint8) * 255
                new_mask = rotate_image(M, mask)
                new_mask = cv2.threshold(new_mask, 127, 255, cv2.THRESH_BINARY)[-1]
                new_masks.update({_id: new_mask})
            return new_masks
        
    class ExtrVecAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.extr_vecs

        def _rotate(self, M:ndarray):
            new_trans_vector_info:dict[int, ndarray] = {}
            angle = np.arccos(M[0,0]) ####
            rot_Z_posture = Posture(rvec=[0, 0, -angle]) # 绕相机坐标系Z轴逆向旋转
            for _id, vecs in self.obj.items():
                org_posture = Posture(rvec=vecs[0], tvec=vecs[1])
                new_posture = rot_Z_posture * org_posture
                new_trans_vector_info.update({_id: np.stack([new_posture.rvec, new_posture.tvec], axis=0)})
            return new_trans_vector_info

    class IntrAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.intr

    class DepthScaleAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.depth_scale

    class Bbox3dAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)
        
        @property
        def obj(self):
            return self.meta.bbox_3d
        
        def _rotate(self, M:cv2.Mat):
            new_bbox_info = {}
            for _id, bbox in self.obj.items():
                new_bbox_info.update({_id: rot_xy_list_2dpoints(M, bbox)})
            return new_bbox_info

    class LandmarksAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.landmarks

        def _rotate(self, M:cv2.Mat):
            new_landmarks_info = {}
            for _id, landmark in self.obj.items():
                new_landmarks_info.update({_id: rot_xy_list_2dpoints(M, landmark)})
            return new_landmarks_info

    class VisibFractAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)
            self.old_masks_callback = lambda :self.get_ap_of_meta(self.meta.MasksAP).obj
            self.new_masks_callback = lambda :self.get_ap_of_meta(self.meta.MasksAP).new_obj

        @property
        def obj(self):
            return self.meta.visib_fract

        def calc_new_visib_fract(self, old_masks:dict[str, np.ndarray], new_masks:dict[str, np.ndarray]):
            # 修改visib_fract
            new_visib_fract = {}
            for _id, visib in self.obj.items():
                if visib == 0:
                    new_visib_fract.update({_id: 0})
                else:
                    mask = old_masks[_id]
                    proj_pixel_num = np.sum(mask.astype(np.bool_)) / visib # 不考虑任何遮挡和相机视野边界的像素数量
                    new_pixel_num = np.sum(new_masks[_id].astype(np.bool_))
                    vf = min(new_pixel_num / proj_pixel_num, 1)
                    new_visib_fract.update({_id: vf})      
            return new_visib_fract 

        def _crop(self, crop_rect: ndarray):
            old_masks = self.old_masks_callback()
            new_masks = self.new_masks_callback()
            return self.calc_new_visib_fract(old_masks, new_masks)
        
        def _rotate(self, crop_rect: ndarray):
            old_masks = self.old_masks_callback()
            new_masks = self.new_masks_callback()
            return self.calc_new_visib_fract(old_masks, new_masks)

    class LabelsAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)
            self.new_masks_callback = lambda :self.get_ap_of_meta(self.meta.MasksAP).new_obj

        @property
        def obj(self):
            return self.meta.labels

        def _crop(self, crop_rect: ndarray):
            new_masks = self.new_masks_callback()
            return calc_bbox2d_from_mask_dict(new_masks)

        def _rotate(self, M:cv2.Mat):
            new_masks = self.new_masks_callback()
            return calc_bbox2d_from_mask_dict(new_masks)

    IGNORE_WARNING = False

    PARA_NAMES = ["color", 
                  "depth", 
                  "masks", 
                  "extr_vecs", 
                  "intr", 
                  "depth_scale", 
                  "bbox_3d", 
                  "landmarks", 
                  "visib_fract", 
                  "labels"]
    
    COLOR = PARA_NAMES[0]
    DEPTH = PARA_NAMES[1]
    MASKS = PARA_NAMES[2]
    EXTR_VECS = PARA_NAMES[3]
    INTR = PARA_NAMES[4]
    DEPTH_SCALE = PARA_NAMES[5]
    BBOX_3D = PARA_NAMES[6]
    LANDMARKS = PARA_NAMES[7]
    VISIB_FRACT = PARA_NAMES[8]
    LABELS = PARA_NAMES[9]

    def __init__(self, 
                color: np.ndarray, 
                depth: np.ndarray, 
                masks: dict[int, np.ndarray], 
                extr_vecs:dict[int, ndarray],
                intr: ndarray,
                depth_scale:float,
                bbox_3d: dict[int, ndarray],        
                landmarks: dict[int, ndarray],
                visib_fract: dict[int, float],
                labels: dict[int, np.ndarray] = None
                # keypoints_visib: dict[int, list],
                ) -> None:
        '''
        color: np.ndarray, 
        depth: np.ndarray, 
        masks: dict[int, np.ndarray], 
        extr_gt: list[dict[int, list]], 外参
        camera: dict, 相机参数
        visib_fract: dict[int, float], 物体可见性
        bbox: dict[int, ndarray:[B, (x,y)]],        
        keypoints: dict[int, ndarray:[N, (x,y)]],
        trans_vector:dict[int, list] 外参
        '''
        ViewMeta.IGNORE_WARNING = True
        self.__init_parameters_keys = [x for x in locals().keys() if x in inspect.signature(self.__init__).parameters.keys()]
        assert ViewMeta.PARA_NAMES == self.__init_parameters_keys, f"ViewMeta.__init__'s parameters must be {ViewMeta.PARA_NAMES}, please check it!"
        self._format_func:dict[str, Callable] = {
            ViewMeta.PARA_NAMES[0]: None,                                                     # color
            ViewMeta.PARA_NAMES[1]: None,                                                     # depth
            ViewMeta.PARA_NAMES[2]: None,                                                     # masks
            ViewMeta.PARA_NAMES[3]: lambda x: self.__reshape_array_in_dict(x, (2, 3)),        # extr_vecs
            ViewMeta.PARA_NAMES[4]: lambda x: np.reshape(x, (3, 3)),                          # intr
            ViewMeta.PARA_NAMES[5]: lambda x: float(x),                                       # depth_scale
            ViewMeta.PARA_NAMES[6]: lambda x: self.__reshape_array_in_dict(x, (-1, 2)),       # bbox_3d
            ViewMeta.PARA_NAMES[7]: lambda x: self.__reshape_array_in_dict(x, (-1, 2)),       # landmarks
            ViewMeta.PARA_NAMES[8]: None,                                                     # visib_fract
            ViewMeta.PARA_NAMES[9]: None,                                                     # labels
        }
        self._agmts_type:dict[str, type] = {
            ViewMeta.PARA_NAMES[0]: ViewMeta.ColorAP,       
            ViewMeta.PARA_NAMES[1]: ViewMeta.DepthAP,     
            ViewMeta.PARA_NAMES[2]: ViewMeta.MasksAP,     
            ViewMeta.PARA_NAMES[3]: ViewMeta.ExtrVecAP,   
            ViewMeta.PARA_NAMES[4]: ViewMeta.IntrAP,      
            ViewMeta.PARA_NAMES[5]: ViewMeta.DepthScaleAP,
            ViewMeta.PARA_NAMES[6]: ViewMeta.Bbox3dAP,    
            ViewMeta.PARA_NAMES[7]: ViewMeta.LandmarksAP, 
            ViewMeta.PARA_NAMES[8]: ViewMeta.VisibFractAP,
            ViewMeta.PARA_NAMES[9]: ViewMeta.LabelsAP    
        }
        self.agmts:dict[str, ViewMeta.AugmentPipeline] = {}
        self.elements = {}
        self.ids = []
        self.color:np.ndarray               = self.set_element(ViewMeta.PARA_NAMES[0], color)
        self.depth:np.ndarray               = self.set_element(ViewMeta.PARA_NAMES[1], depth)
        self.masks:dict[int, ndarray]       = self.set_element(ViewMeta.PARA_NAMES[2], masks)
        self.extr_vecs:dict[int, ndarray]   = self.set_element(ViewMeta.PARA_NAMES[3], extr_vecs)
        self.intr:ndarray                   = self.set_element(ViewMeta.PARA_NAMES[4], intr)
        self.depth_scale: float             = self.set_element(ViewMeta.PARA_NAMES[5], depth_scale)
        self.bbox_3d:dict[int, ndarray]     = self.set_element(ViewMeta.PARA_NAMES[6], bbox_3d)
        self.landmarks:dict[int, ndarray]   = self.set_element(ViewMeta.PARA_NAMES[7], landmarks)
        self.visib_fract: dict[int, float]  = self.set_element(ViewMeta.PARA_NAMES[8], visib_fract)
        self.labels:dict[int, ndarray]      = self.set_element(ViewMeta.PARA_NAMES[9], labels)
        # self.color:np.ndarray               = self.__set_element(ViewMeta.RgbAP,         color)
        # self.depth:np.ndarray               = self.__set_element(ViewMeta.DepthAP,       depth)            
        # self.masks:dict[int, ndarray]       = self.__set_element(ViewMeta.MasksAP,       masks) #[N, H, W]
        # self.extr_vecs:dict[int, ndarray]   = self.__set_element(ViewMeta.ExtrVecAP,     extr_vecs,  lambda x:self.__reshape_array_in_dict(x, (2, 3)))
        # self.intr:ndarray                   = self.__set_element(ViewMeta.IntrAP,        intr,  lambda x: np.reshape(x, (3, 3)))
        # self.depth_scale: float             = self.__set_element(ViewMeta.DepthScaleAP,  depth_scale)
        # self.bbox_3d:dict[int, ndarray]     = self.__set_element(ViewMeta.Bbox3dAP,      bbox_3d,    lambda x:self.__reshape_array_in_dict(x, (-1,2)))
        # self.landmarks:dict[int, ndarray]   = self.__set_element(ViewMeta.LandmarksAP,   landmarks,  lambda x:self.__reshape_array_in_dict(x, (-1,2)))
        # self.visib_fract: dict[int, float]  = self.__set_element(ViewMeta.VisibFractAP,  visib_fract)
        # self.labels:dict[int, ndarray]      = self.__set_element(ViewMeta.LabelsAP,      labels) #[N, H, W]        
        # self.keypoints_visib                = copy.deepcopy(keypoints_visib)
        # self.filter_unvisible()
        ViewMeta.IGNORE_WARNING = False

    def __setattr__(self, __name, __value):
        if not ViewMeta.IGNORE_WARNING:
            warnings.warn("WARNING: Setting properties directly is dangerous and may throw exceptions! make sure you know what you are doing", Warning)
        super().__setattr__(__name, __value)

    @property
    def bbox_2d(self) -> dict[int, np.ndarray]:
        '''
        (x1, y1, x2, y2)
        '''
        if self.labels is not None:
            return self.labels
        elif self.masks is not None:
            ViewMeta.IGNORE_WARNING = True
            self.labels = calc_bbox2d_from_mask_dict(self.masks)
            ViewMeta.IGNORE_WARNING = False
            return self.labels
        return None

    def filter_unvisible(self):
        ids = list(set(self.masks.keys()).union(set(self.visib_fract.keys())))
        for id_ in ids:
            try: mask_cond = np.sum(self.masks[id_]) == 0
            except: mask_cond = False
            try: vf_cond = self.visib_fract[id_] == 0
            except: vf_cond = False
            if mask_cond or vf_cond:
                for value in self.elements.values():
                    if isinstance(value, dict):
                        try:
                            value.pop(id_)
                        except KeyError:
                            pass

    def modify_class_id(self, modify_class_id_pairs:list[tuple[int]]):
        orig_dict_list = get_meta_dict(self)
        modify_class_id(orig_dict_list, modify_class_id_pairs)

    @ignore_viewmeta_warning
    def calc_by_base(self, mesh_dict:dict[int, MeshMeta], cover = False):
        assert self.color is not None
        assert self.extr_vecs is not None
        assert self.intr is not None

        camera_intr = CameraIntr(self.intr, self.color.shape[1], self.color.shape[0], self.depth_scale)
        postures = []
        mesh_list = []
        keys = []
        bbox_3d = {}
        landmarks = {}
        calc_bbox3d     = cover or self.bbox_3d is None
        calc_landmarks  = cover or self.landmarks is None
        for key in self.extr_vecs:
            posture = Posture(rvec=self.extr_vecs[key][0], tvec=self.extr_vecs[key][1])
            meshmeta = mesh_dict[key]
            postures.append(posture)
            mesh_list.append(meshmeta)
            keys.append(key)
            if calc_bbox3d:
                bbox_3d[key] = calc_bbox_3d_proj(meshmeta, posture, camera_intr)
            if calc_landmarks:
                landmarks[key] = calc_landmarks_proj(meshmeta, posture, camera_intr)
        
        if calc_bbox3d:
            self.bbox_3d = bbox_3d
        if calc_landmarks:
            self.landmarks = landmarks

        if cover or self.masks is None or self.visib_fract is None:
            masks, visib_fracts = calc_masks(mesh_list, postures, camera_intr, ignore_depth=True)

            masks_dict = {}
            visib_fract_dict = {}
            for key, mask, visib_fract in zip(keys, masks, visib_fracts):
                masks_dict[key] = mask
                visib_fract_dict[key] = visib_fract

            if cover or self.masks is None:
                self.masks = masks_dict
            if cover or self.visib_fract is None:
                self.visib_fract = visib_fract_dict
        
        self.filter_unvisible()

    @staticmethod
    def from_base_data( color: np.ndarray, 
                        depth: np.ndarray, 
                        extr_vecs:dict[int, ndarray],
                        intr: np.ndarray,
                        depth_scale: float,
                        mesh_dict:dict[int, MeshMeta]):
        viewmeta = ViewMeta(color, depth, None, extr_vecs, intr, depth_scale, None, None, None)
        viewmeta.calc_by_base(mesh_dict)
        return viewmeta

    def set_element(self, name:str, value):
        # pre-process function
        func = self._format_func[name]
        # pre-process to make the data has the correct format
        if func:
            value = func(value) if value is not None else None
        # sort the dict by key, and check if the ids is same
        if isinstance(value, dict):
            value = dict(sorted(value.items(), key=lambda x: x[0]))
            new_ids = list(value.keys())
            if len(self.ids) == 0:
                self.ids = new_ids
            elif self.ids != new_ids:
                raise ValueError("ids must be same")
        self.elements[name] = value

        # set the value to the class
        ViewMeta.IGNORE_WARNING = True
        self.__setattr__(name, value)
        ViewMeta.IGNORE_WARNING = False
        
        return value

    def __set_element(self, ap_type:type, value, func = lambda x:x):
        '''
        ap_type: type of AugmentPipeline
        value: Any
        func: Callable
        '''
        matched_idx = self.__init_parameters_values.index(id(value))
        key = self.__init_parameters_keys[matched_idx]
        self.__init_parameters_keys.pop(matched_idx)
        self.__init_parameters_values.pop(matched_idx)
        # value = func(copy.deepcopy(value))
        value = func(value) if value is not None else None
        if isinstance(value, dict):
            value = dict(sorted(value.items(), key=lambda x: x[0]))
            new_ids = list(value.keys())
            if len(self.ids):
                self.ids = new_ids
            elif self.ids != new_ids:
                raise ValueError("ids must be same")
        self.elements[key] = value
        ### match ap_type with name
        self._agmts_type.update({key: ap_type})
        return value

    @staticmethod
    def __reshape_array_in_dict(dictionary: dict[Any, ndarray], shape):
        if isinstance(dictionary, dict):
            for key in dictionary.keys():
                dictionary[key] = dictionary[key].reshape(shape)
        return dictionary

    def _init_agmts(self):
        # initialize augment pipeline by agmts_type
        for key, ap_type in self._agmts_type.items():
            ap = ap_type(self)
            self.agmts.update({key: ap})        

    def __augment(self, funcname:str, *arg):
        if funcname not in ["crop", "rotate", "change_brightness", "change_saturation"]:
            raise NotImplementedError
        if len(self.agmts) == 0:
            # initialize augment pipeline by agmts_type
            self._init_agmts()
        aug_results = dict(zip(self.agmts.keys(), [agmt.__getattribute__(funcname)(*arg) for agmt in self.agmts.values()]))
        return ViewMeta(**aug_results) 

    ### augment
    def crop(self, crop_rect:np.ndarray):
        '''
        裁剪，去除部分变为全黑，不改变实际图像大小（考虑到裁剪+缩放会导致内参不一致，影响关键点位置预测）
        crop_rect: [N, [y1, x1, y2, x2]]
        '''
        return self.__augment("crop", crop_rect)

    def rotate(self, angle:float):
        '''
        brief
        -----
        旋转

        parameters
        -----
        angle: 旋转角度 弧度制
        
        return
        -----
        Description of the return
        '''
        cam_K = self.intr
        center:tuple[float] = (cam_K[0,2], cam_K[1,2])
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle * 180/ np.pi, 1.0) # 角度制，旋转方向向外，与相机坐标系Z轴相反

        return self.__augment("rotate", M)

    def change_brightness(self, delta_value, direction = (0,0)):
        '''
        brief
        -----
        修改亮度
        
        parameters
        -----
        delta_value: float ,差值
        direction: Iterable,梯度方向，(0,0)则均匀变化 注意是(y,x)
        
        return
        -----
        FieldData
        '''
        direction = np.array(direction, np.float32)
        # 生成新的对象并返回
        return self.__augment("change_brightness", delta_value, direction)

    def change_saturation(self, delta_value):
        '''
        修改饱和度
        '''
        # 生成新的对象并返回
        return self.__augment("change_saturation", delta_value)
    
    def gaussian_noise(self):
        '''
        高斯噪声
        '''
        pass

    def plot(self):
        '''
        显示
        use plt.show() after this method to show
        '''
        plt.subplot(1,2,1)
        if self.masks is not None:
            masks = np.stack(list(self.masks.values()))
            mask = np.sum(masks.astype(np.float32) * 0.2, axis=0).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            # bbox_2d
            for bbox_2d in self.bbox_2d.values():
                plt.gca().add_patch(plt.Rectangle((bbox_2d[0], bbox_2d[1]), 
                                                bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1],
                                                    color="blue", fill=False, linewidth=1))
        else:
            mask = 0
        color = np.clip((self.color.astype(np.float32) + mask), 0, 255).astype(np.uint8)
        plt.imshow(color) 
        # landmarks
        if self.landmarks is not None:
            for ldmks in self.landmarks.values():
                plt.scatter(ldmks[:,0], ldmks[:,1], c = 'g')
        # bbox_3d
        if self.bbox_3d is not None:
            for bbox_3d in self.bbox_3d.values():
                plt.scatter(bbox_3d[:,0], bbox_3d[:,1], c = 'r')
                lines = get_bbox_connections(bbox_3d)
                for line in lines:
                    plt.plot(line[0], line[1], c = 'r')
        # 标注类别与可见性
        if self.bbox_3d is not None:
            for class_id in self.bbox_3d.keys():
                vb = self.visib_fract[class_id]
                label = "{} {:>.3}".format(class_id, float(vb))
                lt = np.min(self.bbox_3d[class_id], axis = 0)
                plt.text(lt[0], lt[1], label, verticalalignment='top')

        plt.subplot(1,2,2)
        if self.depth is not None:
            plt.imshow(self.depth)
            plt.title("depth scale:{}".format(self.depth_scale))

def is_image(array):
    if not isinstance(array, np.ndarray):
        return False
    if array.ndim not in [2, 3]:
        return False
    if array.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        return False
    return True

def serialize_image_container(image_container):
    def serialize_image(image:np.ndarray):  
        # 将NumPy数组编码为png格式的图像
        retval, buffer = cv2.imencode('.png', image)
        # 将图像数据转换为字节字符串
        image_bytes = buffer.tobytes()
        image.tobytes()
        return image_bytes
    if is_image(image_container):
        return serialize_image(image_container)
    elif isinstance(image_container, (list, tuple)):
        new_value = [serialize_image_container(item) for item in image_container]
    elif isinstance(image_container, dict):
        new_value = dict(zip(image_container.keys(), [serialize_image(x) for x in image_container.values()]))
    else:
        return image_container
    return new_value

def deserialize_image_container(bytes_container, imread_flags):
    def deserialize_image(image_bytes):  
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, flags=imread_flags)# 将numpy数组解码为图像
        return image
    if isinstance(bytes_container, bytes):
        return deserialize_image(bytes_container)
    elif isinstance(bytes_container, (list, tuple)):
        new_value = [deserialize_image(item) for item in bytes_container]
    elif isinstance(bytes_container, dict):
        new_value = dict(zip(bytes_container.keys(), [deserialize_image(x) for x in bytes_container.values()]))
    else:
        return bytes_container
    return new_value