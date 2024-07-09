# from https://github.com/ZhiChen902/SC2-PCR

import torch
from SC2PCR.common import knn, rigid_transform_3d
from utils_sc2pcr.SE3 import transform
import json
from easydict import EasyDict as edict
import numpy as np

from .SC2_PCR import Matcher


class SC2PCR:
    def __init__(self) -> None:
        pass

        # construct matcher
        config_path = "SC2PCR/config_KITTI.json"
        config = json.load(open(config_path, 'r'))
        config = edict(config)

        self.matcher = Matcher(inlier_threshold=config.inlier_threshold,
                    num_node=config.num_node,
                    use_mutual=config.use_mutual,
                    d_thre=config.d_thre,
                    num_iterations=config.num_iterations,
                    ratio=config.ratio,
                    nms_radius=config.nms_radius,
                    max_points=config.max_points,
                    k1=config.k1,
                    k2=config.k2)
    
    def forward(self, kpts_src, kpts_dst):
        """
        both: Nx2 with same N
        """
        kpts_src = torch.from_numpy(kpts_src[None, ...]).to(torch.float32)
        kpts_dst = torch.from_numpy(kpts_dst[None, ...]).to(torch.float32)
        trans = self.matcher.SC2_PCR(kpts_src, kpts_dst)[0].numpy()
        return trans

