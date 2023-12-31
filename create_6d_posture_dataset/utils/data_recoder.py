import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from posture_6d.dataset_format import DatasetFormatMode
from posture_6d.dataset_format import DatasetFormat, Elements
from .. import RGB_DIR, DEPTH_DIR, TRANS_DIR

class FrameMeta():
    def __init__(self, trans_mat_Cn2C0, rgb = None, depth = None) -> None:
        self.trans_mat_Cn2C0 = trans_mat_Cn2C0
        self.color = rgb
        self.depth = depth

class Recoder(DatasetFormat):
    def __init__(self, directory, std_models_dir, clear_incomplete=False, init_mode=DatasetFormatMode.NORMAL) -> None:
        super().__init__(directory, clear_incomplete, init_mode)
        self.std_models_dir = std_models_dir
        std_models_names = [os.path.splitext(x)[0] for x in os.listdir(self.std_models_dir) if os.path.splitext(x)[1] == ".ply"] # 获取标准模型的名称
        self.std_models_names:list = std_models_names
        self.std_models_names.insert(0, "global_base_frames") # 在标准模型列表的第一位插入"global_base_frames"
        self.std_models_names.insert(1, "local_base_frames") # 在标准模型列表的第二位插入"local_base_frames"
        self.std_models_names.append("dataset_frames") # 在标准模型列表的最后一位插入"dataset_frames"

        self.categroy_index = 0
        self.AddNum = 0 # 当前标准模型已采集的增量帧数
        self.skip_segs = []
    
    def _init_clusters(self):
        super()._init_clusters()
        self.close_all()
        self.rgb_elements   = Elements(self,      RGB_DIR,
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.depth_elements = Elements(self,      DEPTH_DIR,    
                                       read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.trans_elements = Elements(self,    TRANS_DIR,
                                       read_func=np.load,
                                       write_func=np.save,
                                       suffix='.npy')

    def inc_idx(self):
        self.categroy_index += 1
        self.categroy_index = min(self.categroy_index, len(self.std_models_names)-1)
        self.AddNum = 0

    def dec_idx(self):
        self.categroy_index -= 1
        self.categroy_index = max(self.categroy_index, 0)
        self.AddNum = 0

    def clear_skip_segs(self):
        self.skip_segs.clear()

    def add_skip_seg(self, seg):
        if len(self.skip_segs) > 0:
            if seg <= max(self.skip_segs):
                return
            elif seg > len(self.std_models_names):
                return
        self.skip_segs.append(seg)

    def skip_to_seg(self):
        '''
        跳过，直到分段点
        '''
        try:
            skip_to = self.skip_segs.pop(0)
            skip_to = min(skip_to, len(self.std_models_names))
            skip_to = max(skip_to, self.categroy_index)
        except:
            skip_to = len(self.std_models_names)
        while self.categroy_index != skip_to:
            self.inc_idx()

    def is_all_recorded(self):
        return self.categroy_index >= len(self.std_models_names)

    def read_one(self, data_i) -> FrameMeta:
        super().read_one(data_i)
        self.rgb_elements._init_data_i_dir_map()
        self.depth_elements._init_data_i_dir_map()
        self.trans_elements._init_data_i_dir_map()

        rgb = self.rgb_elements.read(data_i)
        depth = self.depth_elements.read(data_i)
        trans = self.trans_elements.read(data_i)

        return FrameMeta(trans_mat_Cn2C0=trans, rgb=rgb, depth=depth)

    def write_element(self, framemeta: FrameMeta, data_i: int):
        appdir = self.std_models_names[self.categroy_index]
        self.rgb_elements.write(data_i,    framemeta.color,   appdir=appdir)
        self.depth_elements.write(data_i,  framemeta.depth, appdir=appdir)
        self.trans_elements.write(data_i,  framemeta.trans_mat_Cn2C0, appname=appdir)
        ### ADD    
        self.AddNum += 1

    def write_to_disk(self, framemeta: FrameMeta, data_i=-1):
        self._updata_data_num()        
        if data_i == -1:
            data_i = self.data_num
        return self.write_element(framemeta, data_i)

    def save_frames(self, c, d, t):
        framemeta = FrameMeta(t, c, d)
        self.write_to_disk(framemeta)

    def delete(self, delete_list:list, change_file = True):
        pass

    def insert(self, insert_list:list, change_file = True):
        pass

    def rename_all(self, exchange_pair = []):
        pass

    def make_directories(self):
        pass
        # if os.path.exists(self.rgb_dir):
        #     return
        # else:
        #     for d in [self.rgb_dir, self.depth_dir, self.trans_dir]:
        #         try:
        #             shutil.rmtree(d)
        #         except:
        #             pass
        #         os.makedirs(d)
        #     with open(self.directory+'model_index_range.json', 'w') as fp:
        #         json.dump(self.model_index_dict, fp)