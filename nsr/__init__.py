# triplane, tensorF etc.
from .train_util import TrainLoop3DRec, TrainLoop3DRecTrajVis
from .train_util_cvD import TrainLoop3DcvD

# train ffhq
from .cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD, TrainLoop3DcvD_nvsD_canoD_eg3d
from .train_util_with_eg3d import TrainLoop3DRecEG3D
# from .train_util_with_eg3d_real import TrainLoop3DRecEG3DReal, TrainLoop3DRecEG3DRealOnly
# from .train_util_with_eg3d_real_D import TrainLoop3DRecEG3DRealOnl_D

# * difffusion trainer
from .train_util_diffusion import TrainLoop3DDiffusion

# import lsgm
from .lsgm import *
from .lsgm import crossattn_cldm_objv