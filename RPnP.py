import numpy as np
import cv2

class Res():
    def __init__(self, R, t, r) -> None:
        self.R = R
        self.t = t
        self.r = r

def solveRPnP(objectPoints: np.ndarray,
    imagePoints: np.ndarray,
    cameraMatrix: np.ndarray):
    '''
    brief
    -----
    RPnP方法，源码参考https://github.com/xuchi7/RPnP
    
    parameters
    -----
    objectPoints: np.ndarray:   三维点[N, 3]
    imagePoints: np.ndarray:    二维点[N, 2]
    cameraMatrix: np.ndarray:   相机内参
    
    return
    -----
    Description of the return
    '''

    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]

    imagePoints[:,0] = (imagePoints[:,0] - cx)/fx
    imagePoints[:,1] = (imagePoints[:,1] - cy)/fy
    [success, R, tvec] = RPnP(objectPoints, imagePoints)
    rvec = cv2.Rodrigues(R)[0][:,0]
    return success, rvec, tvec

    

def RPnP(point_3d:np.ndarray, point_2d:np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]: 
    R = np.eye(3)
    t = np.array([0,0,0])
    n = len(point_2d)    
    point_3d = point_3d.T
    point_2d = point_2d.T    

    XXw = point_3d
    xxv = np.vstack([point_2d,np.ones((1,n))])
    xxv = (xxv / np.linalg.norm(xxv, axis=0))
    
    # selecting an edge $P_{i1}P_{i2}$ by n random sampling
    # 源代码是找投影距离最近的两点，但文档说法是找最远的两点
    i1 = 0
    i2 = 1
    lmin = np.linalg.norm(xxv[i1, :] - xxv[i2, :])
    rij = np.ceil(np.random.rand(n,2) * n)
    for ii in np.arange(n):
        i = rij[ii,0]
        j = rij[ii,1]
        if i == j:
            continue
        l = np.linalg.norm(xxv[i1, :] - xxv[i2, :])
        if l < lmin:
            i1 = i
            i2 = j
            lmin = l

    # calculating the rotation matrix of $O_aX_aY_aZ_a$.
    p1 = point_3d[:,i1]
    p2 = point_3d[:,i2]
    p0 = (p1 + p2) / 2
    x = p2 - p0
    x = x / np.linalg.norm(x)
    if np.abs(np.array([0,1,0]).dot(x)) < np.abs(np.array([0,0,1]).dot(x)):
        z = np.cross(x, np.array([0,1,0]))
        z = z / np.linalg.norm(z)
        y = np.cross(z,x)
        y = y / np.linalg.norm(y)
    else:
        y = np.cross(np.array([0,0,1]),x)
        y = y / np.linalg.norm(y)
        z = np.cross(x,y)
        z = z / np.linalg.norm(z)
    
    Ro = np.vstack([x,y,z]).T
    # transforming the reference points form orignial object space
# to the new coordinate frame  $O_aX_aY_aZ_a$.
    point_3d = (Ro.T).dot(point_3d - np.tile(np.expand_dims(p0, -1), [1, n]))
    # Dividing the n-point set into (n-2) 3-point subsets
# and setting up the P3P equations
    
    v1 = xxv[:,i1]
    v2 = xxv[:,i2]
    cg1 = np.transpose(v1).dot(v2)
    sg1 = np.sqrt(1 - cg1 ** 2)
    D1 = np.linalg.norm(point_3d[:,i1] - point_3d[:,i2])
    D4 = np.zeros((n - 2,5))
    if 0:
        j = 0
        for i in range(n):
            if i == i1 or i == i2:
                continue
            j = j + 1
            vi = xxv[:,i]
            cg2 = np.transpose(v1) * vi
            cg3 = np.transpose(v2) * vi
            sg2 = np.sqrt(1 - cg2 ** 2)
            D2 = np.linalg.norm(point_3d[:,i1] - point_3d[:,i])
            D3 = np.linalg.norm(point_3d[:,i] - point_3d[:,i2])
            # get the coefficients of the P3P equation from each subset.
            D4[j,:] = getp3p(cg1,cg2,cg3,sg1,sg2,D1,D2,D3)
        # get the 7th order polynomial, the deviation of the cost function.
        D7 = np.zeros((1,8))
        for i in np.arange(1,n - 2+1).reshape(-1):
            D7 = D7 + getpoly7(D4[i,:])
    else:
        # following code is the same as the code above (between "if 0" and "else")
        # but the following code is a little more efficient than the former
        # in matlab when the number of points is large,
        # because the dot multiply operation is used.
        idx = np.ones(n, np.bool_)
        idx[[i1,i2]] = False
        vi = xxv[:,idx]
        cg2 = np.transpose(vi).dot(v1)
        cg3 = np.transpose(vi).dot(v2)
        sg2 = np.sqrt(1 - np.square(cg2))
        D2 = cg2.copy()
        D3 = cg2.copy()
        didx = np.where(idx)[0]
        for i in range(n - 2):
            D2[i] = np.linalg.norm(point_3d[:,didx[i]] - point_3d[:,i1])
            D3[i] = np.linalg.norm(point_3d[:,didx[i]] - point_3d[:,i2])
        A1 = (D2 / D1) ** 2
        A2 = A1 * sg1 ** 2 - sg2 ** 2
        A3 = np.multiply(cg2,cg3) - cg1
        A4 = cg1 * cg3 - cg2
        A6 = (D3 ** 2 - D1 ** 2 - D2 ** 2) / (2 * D1 ** 2)
        A7 = 1 - cg1 ** 2 - cg2 ** 2 + np.multiply(cg1 * cg2,cg3) + np.multiply(A6,sg1 ** 2)
        D4 = np.array([A6 ** 2 - np.multiply(A1,cg3 ** 2),
                       2 * (np.multiply(A3,A6) - np.multiply(np.multiply(A1,A4),cg3)),
                       A3 ** 2 + np.multiply(2 * A6,A7) - np.multiply(A1,A4 ** 2) - np.multiply(A2,cg3 ** 2),
                       2 * (np.multiply(A3,A7) - np.multiply(np.multiply(A2,A4),cg3)),
                       A7 ** 2 - np.multiply(A2,A4 ** 2)]).T
        F7 = np.array([4 * D4[:,0] ** 2, 
                       np.multiply(7 * D4[:,1],D4[:,0]),
                       np.multiply(6 * D4[:,2],D4[:,0]) + 3 * D4[:,1] ** 2,
                       np.multiply(5 * D4[:,3],D4[:,0]) + np.multiply(5 * D4[:,2],D4[:,1]),
                       np.multiply(4 * D4[:,4],D4[:,0]) + np.multiply(4 * D4[:,3],D4[:,1]) + 2 * D4[:,2] ** 2,
                       np.multiply(3 * D4[:,4],D4[:,1]) + np.multiply(3 * D4[:,3],D4[:,2]),
                       np.multiply(2 * D4[:,4],D4[:,2]) + D4[:,3] ** 2,
                       np.multiply(D4[:,4],D4[:,3])]).T
        D7 = sum(F7)
    
    # retriving the local minima of the cost function.
    t2s = np.roots(D7)
    maxreal = np.amax(np.abs(np.real(t2s)))
    t2s = t2s[np.logical_not((np.abs(np.imag(t2s)) / maxreal) > 0.001)]
    t2s = np.real(t2s)
    D6 = np.arange(7,0,- 1) * D7[:7]
    F6 = np.polyval(D6,t2s)
    t2s = t2s[np.logical_not(F6 <= 0)]
    if len(t2s)==0:
        print('no solution!\n' % ())
        return False, R, t
    
    # calculating the camera pose from each local minimum.
    m = len(t2s)
    res = []
    for i in range(m):
        t2 = t2s[i]
        # calculating the rotation matrix
        d2 = cg1 + t2
        x = v2 * d2 - v1
        x = x / np.linalg.norm(x)
        if np.abs(np.array([0,1,0]).dot(x)) < np.abs(np.array([0,0,1]).dot(x)):
            z = np.cross(x,np.array([0,1,0]))
            z = z / np.linalg.norm(z)
            y = np.cross(z,x)
            y = y / np.linalg.norm(y)
        else:
            y = np.cross(np.array([0,0,1]),x)
            y = y / np.linalg.norm(y)
            z = np.cross(x,y)
            z = z / np.linalg.norm(z)
        Rx = np.array([x,y,z]).T
        # calculating c, s, tx, ty, tz
        D = np.zeros((2 * n,6))
        r = Rx.reshape(-1)
        for j in range(n):
            ui = point_2d[0,j]
            vi = point_2d[1,j]
            xi = point_3d[0,j]
            yi = point_3d[1,j]
            zi = point_3d[2,j]
            D[2 * j,:] = np.array([- r[1] * yi + ui * (r[7] * yi + r[8] * zi) - r[2] * zi,
                                       - r[2] * yi + ui * (r[8] * yi - r[7] * zi) + r[1] * zi,
                                       - 1,
                                       0,
                                       ui,
                                       ui * r[6] * xi - r[0] * xi])
            D[2 * j + 1,:] = np.array([- r[4] * yi + vi * (r[7] * yi + r[8] * zi) - r[5] * zi,
                                   - r[5] * yi + vi * (r[8] * yi - r[7] * zi) + r[4] * zi,
                                   0,
                                   - 1,
                                   vi,
                                   vi * r[6] * xi - r[3] * xi])
        DTD = np.transpose(D).dot(D)
        D, V = np.linalg.eig(DTD)
        D = D[::-1]
        V = V[:,::-1]
        V1 = V[:, 0]
        V1 = V1 / V1[-1]
        c = V1[0]
        s = V1[1]
        t = V1[2:5]
        # calculating the camera pose by 3d alignment
        xi = point_3d[0,:]
        yi = point_3d[1,:]
        zi = point_3d[2,:]
        XXcs = np.array([r[0] * xi + (r[1] * c + r[2] * s) * yi + (- r[1] * s + r[2] * c) * zi + t[0],
                         r[3] * xi + (r[4] * c + r[5] * s) * yi + (- r[4] * s + r[5] * c) * zi + t[1],
                         r[6] * xi + (r[7] * c + r[8] * s) * yi + (- r[7] * s + r[8] * c) * zi + t[2]])
        XXc = np.zeros(XXcs.shape)
        for j in range(n):
            XXc[:,j] = xxv[:,j] * np.linalg.norm(XXcs[:,j])
        R,t = calcampose(XXc,XXw)
        # calculating the reprojection error
        XXc = R.dot(XXw) + np.expand_dims(t, -1).dot(np.ones((1,n)))
        xxc = np.array([[XXc[0,:] / XXc[2,:]],[XXc[1,:] / XXc[2,:]]])
        r = np.mean(np.sqrt(sum((xxc - point_2d) ** 2)))
        
        res.append(Res(R, t, r))
    
    # determing the camera pose with the smallest reprojection error.
    minr = np.inf
    for i in range(m):
        if res[i].r < minr:
            minr = res[i].r
            R = res[i].R
            t = res[i].t
    
    return True, R,t
    
def getp3p(l1:np.ndarray, l2:np.ndarray, A5:np.ndarray, C1:np.ndarray, C2:np.ndarray, D1:float, D2:float, D3:float): 
    A1 = (D2 / D1) ** 2
    A2 = A1 * C1 ** 2 - C2 ** 2
    A3 = l2 * A5 - l1
    A4 = l1 * A5 - l2
    A6 = (D3 ** 2 - D1 ** 2 - D2 ** 2) / (2 * D1 ** 2)
    A7 = 1 - l1 ** 2 - l2 ** 2 + l1 * l2 * A5 + A6 * C1 ** 2
    B = np.array([A6 ** 2 - A1 * A5 ** 2,2 * (A3 * A6 - A1 * A4 * A5),A3 ** 2 + 2 * A6 * A7 - A1 * A4 ** 2 - A2 * A5 ** 2,2 * (A3 * A7 - A2 * A4 * A5),A7 ** 2 - A2 * A4 ** 2])
    return B
    
def getpoly7(F:np.ndarray): 
    F7 = np.transpose(np.array([[4 * F[0] ** 2],[7 * F[1] * F[0]],[6 * F[2] * F[0] + 3 * F[1] ** 2],[5 * F[3] * F[0] + 5 * F[2] * F[1]],[4 * F[4] * F[0] + 4 * F[3] * F[1] + 2 * F[2] ** 2],[3 * F[4] * F[1] + 3 * F[3] * F[2]],[2 * F[4] * F[2] + F[3] ** 2],[F[4] * F[3]]]))
    return F7
    
def calcampose(XXc:np.ndarray, XXw:np.ndarray): 
    n = np.shape(XXc)[1]
    X = XXw
    Y = XXc
    K = np.eye(n) - np.ones((n,n)) / n
    ux = np.mean(X,1)
    uy = np.mean(Y,1)
    sigmx2 = np.mean(sum((X.dot(K)) ** 2))
    SXY = np.linalg.multi_dot((Y, K, X.T))/ n
    (U, D, V) = np.linalg.svd(SXY)
    D = np.array([[D[0], 0, 0], 
                  [0, D[1], 0],
                  [0, 0, D[2]]])
    S = np.eye(3)
    if np.linalg.det(SXY) < 0:
        S[2,2] = - 1
    
    R2 = np.linalg.multi_dot((U, S, V))
    c2 = np.trace(D.dot(S)) / sigmx2
    t2 = uy - c2 * R2.dot(ux)
    X = R2[:,0]
    Y = R2[:,1]
    Z = R2[:,2]
    if np.linalg.norm(np.cross(X,Y) - Z) > 0.02:
        R2[:,2] = - Z
    
    return R2,t2
    
if __name__ == "__main__":
    point_2d = np.loadtxt("point_2d.txt")
    point_3d = np.loadtxt("point_3d.txt")
    point_gt = np.loadtxt("gt_kp.txt") 
    K = np.loadtxt("K.txt")
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    point_2d[:,0] = (point_2d[:,0] - cx)/fx
    point_2d[:,1] = (point_2d[:,1] - cy)/fy
    success, R, t = RPnP(point_3d, point_2d)