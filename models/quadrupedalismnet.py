import pickle
import torch.nn as nn
from .poseverts import verts_core, posemap, MatVecMult
import numpy as np
import chumpy as ch

class SMAL(object):
  def __init__(self, model_path, cfg):
    self.cfg = cfg

    dd = pickle.load(open(model_path, 'rb'), encoding='latin1')
    self.f = dd['f']
    self.v = self.get_template(self.cfg)

  def get_template(self, cfg):
    model = self.load_model(cfg['MODEL_PATH'])
    nBetas = len(model.betas.r)
    data = pickle.load(open(cfg['DATA_PATH'], 'rb'), encoding='latin1')
    betas = data['cluster_means'][2][:nBetas]
    print(betas.shape)

  def load_model(self, fname_or_dict):
    dd = self.ready_arguments(fname_or_dict)
    
    for i in dd:
        print(i)

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

    for i in dd:
        print(i)
        if isinstance(dd[i], list) or isinstance(dd[i], str):   
            print(len(dd[i]))
        else:
            print(dd[i].shape)

    print('\n')
        
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

class QuadrupedalismNet(nn.Module):

  def __init__(self, input_shape, cfg):
    super(QuadrupedalismNet, self).__init__()
    self.cfg = cfg
    self.smal = SMAL(self.cfg['MODEL_PATH'], cfg)




