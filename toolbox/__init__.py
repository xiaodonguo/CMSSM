from .metrics_CART import averageMeter, runningScore
from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in [ 'pst900', 'CART']


    if cfg['dataset'] == 'CART':
        from .datasets.wild import Wild
        # return SUS(cfg, mode='trainval'), SUS(cfg, mode='test')
        return Wild(cfg, mode='train'), Wild(cfg, mode='val'), Wild(cfg, mode='test')

    if cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PST900

        return PST900(cfg, mode='train'), PST900(cfg, mode='test')




def get_model(cfg):

    ############# model_others ################


    if cfg['model_name'] == 'b1_add':
        from models.CM_SSM import Model
        return Model(mode='b1', n_class=12, inputs='rgbt', fusion_mode='add')



    # Proposed models
    if cfg['model_name'] == 'b1_CM-SSM':
        from models.CM_SSM import Model
        return Model(mode='b1', n_class=cfg['n_classes'], inputs=cfg['inputs'], fusion_mode='CM-SSM')





