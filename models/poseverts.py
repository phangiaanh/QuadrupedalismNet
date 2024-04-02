import chumpy as ch
import numpy as np
import cv2
import scipy.sparse as sp
from chumpy.ch import MatVecMult

class Rodrigues(ch.Ch):
    dterms = 'rt'
    
    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p): 
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()        
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1,3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp)-ch.eye(3)).ravel() for pp in p]).ravel()

def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))



def ischumpy(x): return hasattr(x, 'dterms')

def verts_decorated(trans, pose, 
    v_template, J, weights, kintree_table, bs_style, f,
    bs_type=None, posedirs=None, betas=None, shapedirs=None, want_Jtr=False):

    for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
        if which is not None:
            assert ischumpy(which)

    v = v_template

    if shapedirs is not None:
        if betas is None:
            betas = ch.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v
        
    if posedirs is not None:
        v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
    else:
        v_posed = v_shaped
        
    v = v_posed
        
    if sp.issparse(J):
        regressor = J
        J_tmpx = MatVecMult(regressor, v_shaped[:,0])        
        J_tmpy = MatVecMult(regressor, v_shaped[:,1])        
        J_tmpz = MatVecMult(regressor, v_shaped[:,2])        
        J = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T            
    else:    
        assert(ischumpy(J))
        
    assert(bs_style=='lbs')
    result, Jtr = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr=True, xp=ch)
     
    tr = trans.reshape((1,3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type =bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    return result

def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=ch):
    
    if xp == ch:
        assert(hasattr(pose, 'dterms'))
        assert(hasattr(v, 'dterms'))
        assert(hasattr(J, 'dterms'))
        assert(hasattr(weights, 'dterms'))
     
    assert(bs_style=='lbs')
    result = verts_core_lbs(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result

def global_rigid_transformation(pose, J, kintree_table, xp):
    results = {}
    pose = pose.reshape((-1,3))
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}
    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    if xp == ch:
        rodrigues = lambda x : Rodrigues(x)
    else:
        import cv2
        rodrigues = lambda x : cv2.Rodrigues(x)[0]

    with_zeros = lambda x : xp.vstack((x, xp.array([[0.0, 0.0, 0.0, 1.0]])))
    results[0] = with_zeros(xp.hstack((rodrigues(pose[0,:]), J[0,:].reshape((3,1)))))        
        
    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(xp.hstack((
            rodrigues(pose[i,:]),
            ((J[i,:] - J[parent[i],:]).reshape((3,1)))
            ))))

    pack = lambda x : xp.hstack([np.zeros((4, 3)), x.reshape((4,1))])
    
    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        results2 = [results[i] - (pack(
            results[i].dot(xp.concatenate( ( (J[i,:]), 0 ) )))
            ) for i in range(len(results))]
        results = results2
    result = xp.dstack(results)
    return result, results_global


def verts_core_lbs(pose, v, J, weights, kintree_table, want_Jtr=False, xp=ch):
    A, A_global = global_rigid_transformation(pose, J, kintree_table, xp)
    T = A.dot(weights.T)

    rest_shape_h = xp.vstack((v.T, np.ones((1, v.shape[0]))))
        
    v =(T[:,0,:] * rest_shape_h[0, :].reshape((1, -1)) + 
        T[:,1,:] * rest_shape_h[1, :].reshape((1, -1)) + 
        T[:,2,:] * rest_shape_h[2, :].reshape((1, -1)) + 
        T[:,3,:] * rest_shape_h[3, :].reshape((1, -1))).T

    v = v[:,:3] 
    
    if not want_Jtr:
        return v
    Jtr = xp.vstack([g[:3,3] for g in A_global])
    return (v, Jtr)