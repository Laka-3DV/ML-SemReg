import os
import open3d as o3d
import numpy as np

data_names = [
    "pts_src", 
    "label_src", 
    "kpts_src", 
    "gedi_src", 
    "pts_dst", 
    "label_dst", 
    "kpts_dst", 
    "gedi_dst", 
    "trans_gt",
]

class DataLoader:
    def __init__(self) -> None:
        self.len = 40
    
    def __len__(self):
        return self.len
    
    def get_item(self, idx):
        assert idx < self.len
        # todo
        data_load = np.load(file="./data/{:06}.npz".format(idx))
        data_list = [data_load[name] for name in data_names]
        return data_list

def get_cmd_config():
    from argparse import ArgumentParser
    parse = ArgumentParser()
    parse.add_argument("-is_vis", default=False, action="store_true")
    return parse.parse_args()

def vis_pcr_pcd(pts_src, pts_dst, trans):
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pts_src)

    pcd_src.paint_uniform_color([1, 0.706, 0])
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(pts_dst)
    pcd_dst.paint_uniform_color([0, 0.651, 0.929])


    o3d.visualization.draw_geometries([pcd_src, pcd_dst])

    pcd_src.transform(trans)
    o3d.visualization.draw_geometries([pcd_src, pcd_dst])

