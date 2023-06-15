import numpy as np
import cv2

class Posture:
    '''
    姿态类，接收各种类型的姿态的输入，并转换为矩阵
    '''
    POSTURE_VEC = 0
    POSTURE_MAT = 1
    POSTURE_HOMOMAT = 2
    POSTURE_EULARZYX = 3
    
    def __init__(self, *, rvec:np.ndarray = None, tvec:np.ndarray = None,
                                        rmat:np.ndarray = None,
                                        homomat:np.ndarray = None,
                                        EularZYX:np.ndarray = None
                                        ) -> None:
        self.trans_mat:np.ndarray  = np.eye(4)
        if rvec is not None:
            if isinstance( rvec, (list, tuple)):
                rvec = np.array(rvec, np.float32).squeeze()
            self.set_rvec(rvec)
            # if tvec is not None:
            #     self.set_tvec(tvec.squeeze())
            # else:
            #     self.set_tvec(np.array([0., 0., 0.]))
        elif rmat is not None:
            self.set_rmat(rmat)
            if tvec is not None:
                self.set_tvec(tvec)
            else:
                self.set_tvec(np.array([0., 0., 0.]))
        elif tvec is not None:
            if isinstance(tvec, (list, tuple)):
                tvec = np.array(tvec, np.float32).squeeze()
            self.set_tvec(tvec)
        elif homomat is not None:
            self.set_homomat(homomat)
        else:
            pass

    def __mul__(self, posture):
        posture = Posture(homomat = self.trans_mat.dot(posture.trans_mat))
        return posture
    
    @property
    def inv_transmat(self) -> np.ndarray :
        return np.linalg.inv(self.trans_mat)
    
    @property
    def rvec(self) -> np.ndarray :
        return cv2.Rodrigues(self.trans_mat[:3,:3])[0][:,0]
    
    @property
    def tvec(self) -> np.ndarray :
        return self.trans_mat[:3,3].T
    
    @property
    def rmat(self) -> np.ndarray :
        return self.trans_mat[:3,:3]
    
    @property
    def eularZYX(self):
        pass
    
    def set_rvec(self, rvec):
        self.trans_mat[:3,:3] = cv2.Rodrigues(rvec)[0]

    def set_tvec(self, tvec):
        self.trans_mat[:3,3] = tvec

    def set_rmat(self, rmat):
        self.trans_mat[:3,:3] = rmat

    def set_homomat(self, homomat):
        self.trans_mat:np.ndarray = homomat.copy()

if __name__ == "__main__":
    Posture()