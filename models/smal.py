import pickle
from .poseverts import verts_core, posemap, MatVecMult, batch_rodrigues, batch_global_rigid_transformation
import numpy as np
import chumpy as ch
import os
import torch

def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(object):
  def __init__(self, model_path, cfg):
    self.cfg = cfg

    dd = pickle.load(open(model_path, 'rb'), encoding='latin1')
    self.f = dd['f']
    self.v_template = self.get_template(self.cfg)
    v, self.left_inds, self.right_inds, self.center_inds = self.align_smal_template_to_symmetry_axis()

    if torch.cuda.is_available():
      self.v_template = torch.Tensor(self.v_template).cuda()
    else:
      self.v_template = torch.Tensor(self.v_template)

    self.size = [self.v_template.shape[0], 3]
    self.num_betas = dd['shapedirs'].shape[-1]


    # Shape basis
    shapedir = torch.tensor(np.reshape(undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T)
    if torch.cuda.is_available():
      self.shapedirs = torch.tensor(shapedir, dtype=torch.float32).cuda()
    else:
      self.shapedirs = torch.tensor(shapedir, dtype=torch.float32)


    # Regressor of joint locations
    if torch.cuda.is_available():
      self.J_regressor = torch.Tensor(dd['J_regressor'].T.todense()).cuda()
    else:
      self.J_regressor = torch.Tensor(dd['J_regressor'].T.todense())

    # Pose basis
    num_pose_basis = dd['posedirs'].shape[-1]
    print(f"asasa {dd['posedirs'].shape}")
    posedirs = np.reshape(undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
    if torch.cuda.is_available():
      self.posedirs = torch.Tensor(posedirs).cuda()
    else:
      self.posedirs = torch.Tensor(posedirs)

    
    self.parents = dd['kintree_table'][0].astype(np.int32)
    if torch.cuda.is_available():
        self.weights = torch.Tensor(undo_chumpy(dd['weights'])).cuda()
    else:
        self.weights = torch.Tensor(undo_chumpy(dd['weights']))


  def __call__(self, beta, theta, trans=None, del_v=None, get_skin=True):

    # 1. Add shape blend shapes
    nBetas = beta.shape[1]
    print(self.v_template.dtype)
    print(beta.dtype)
    print(self.shapedirs.dtype)
    if del_v is None:
      v_shaped = self.v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
    else:
      v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])

    
    # 2. Infer shape-dependent joint locations.
    print(v_shaped.shape)
    Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
    Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
    Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
    J = torch.stack([Jx, Jy, Jz], dim=2)


    # 3. Add pose blend shapes
    Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3]), opts=self.cfg), [-1, 35, 3, 3])
    # Ignore global rotation.
    print(Rs[:, 1:, :, :].shape)
    if torch.cuda.is_available():
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).cuda(), [-1, 306])
    else:
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3), [-1, 306])

    v_posed = torch.reshape(torch.matmul(pose_feature, self.posedirs), [-1, self.size[0], self.size[1]]) + v_shaped


    # 4. Get global joint locations
    self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, opts=self.cfg)


    # 5. Skinning
    num_batch = theta.shape[0]
        
    weights_t = self.weights.repeat([num_batch, 1])
    W = torch.reshape(weights_t, [num_batch, -1, 35])
        
    T = torch.reshape(torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])), [num_batch, -1, 4, 4])
    
    if torch.cuda.is_available():
        v_posed_homo = torch.cat([v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).cuda()], 2)
    else:
        v_posed_homo = torch.cat([v_posed, torch.ones([num_batch, v_posed.shape[1], 1])], 2)
    
    
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]
    if trans is None:
        if torch.cuda.is_available():
            trans = torch.zeros((num_batch,3)).cuda()
        else:
            trans = torch.zeros((num_batch,3))


    verts = verts + trans[:,None,:]
    # Get joints:
    joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
    joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
    joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
    joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
    if get_skin:
        return verts, joints, Rs
    else:
        return joints





  def get_template(self, cfg):
    model = self.load_model(cfg['MODEL_PATH'])
    nBetas = len(model.betas.r)
    print("aaa")
    print(nBetas)
    data = pickle.load(open(cfg['DATA_PATH'], 'rb'), encoding='latin1')
    betas = data['cluster_means'][:nBetas]
    # model.betas[:] = betas
    # print(betas.shape)
    # print(model.betas.shape)
    v = model.r.copy()
    return v

  def load_model(self, fname_or_dict):
    dd = self.ready_arguments(fname_or_dict)

    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style']
    }
    
    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1,3))
    result.J_transformed = Jtr + dd['trans'].reshape((1,3))

    for k, v in dd.items():
        setattr(result, k, v)
        
    return result

  def ready_arguments(self, fname_or_dict):

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        dd = fname_or_dict

    # for i in dd:
    #     print(i)
    #     if isinstance(dd[i], list) or isinstance(dd[i], str):   
    #         print(len(dd[i]))
    #     else:
    #         print(dd[i].shape)

        
    self.backwards_compatibility_replacements(dd)
        
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:,0])        
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:,1])        
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:,2])        
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T    
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:    
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
            
    return dd

  def backwards_compatibility_replacements(self, dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'

  def align_smal_template_to_symmetry_axis(self):
    # These are the indexes of the points that are on the symmetry axis
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]

    v = self.v_template
    v = v - np.mean(v)
    y = np.mean(v[I,1])
    v[:,1] = v[:,1] - y
    v[I,1] = 0
    
    sym_path = os.path.join(self.cfg['GEN_DIR'], 'symIdx.pkl')
    symIdx = pickle.load(open(sym_path, 'rb'), encoding='latin1')
    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    v[left[symIdx]] = np.array([1,-1,1])*v[left]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert(len(left_inds) == len(right_inds))
    except:
        import pdb; pdb.set_trace()

    return v, left_inds, right_inds, center_inds
