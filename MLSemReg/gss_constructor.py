import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import eval_taichi
from utils.voxel_downsample import voxel_downsample_with_label


def extract_init_guidpost_outdoor(
    pts, labels,
    label_usefull,
    is_vis=False,
    **kwargs
):
    # extrct usefull pts
    indic = np.any(labels[:, None] == label_usefull[None, :], axis=1)
    pts = pts[indic]
    labels = labels[indic]

    if len(pts) == 0:
        return None, None

    label_unique, counts_unique = np.unique(labels, return_counts=True)
    label_unique = label_unique[counts_unique >= 5]
    counts_unique = counts_unique[counts_unique >= 5]

    # label -> cluster -> pts
    guidpost_centers = {}  # label: pts
    dbscaner = DBSCAN(eps=1.5, min_samples=3)
    for label_gd in label_usefull:
        if label_gd not in label_unique:
            continue

        if is_vis:
            print(label_gd)

        # Nx3 points with same label `label_gt`
        pts_group = pts[labels == label_gd]

        # cluster -> centers in xy-plane
        cluster_labels = dbscaner.fit_predict(
            pts_group[:, :2])  

        cluster_label_unique = np.unique(cluster_labels)
        clu_centers = []
        geo_list = []
        for cluster_l in cluster_label_unique: 
            if cluster_l == -1:
                continue

            pts_cluster = pts_group[cluster_labels == cluster_l]  
            cluster_center = np.mean(pts_cluster, axis=0)

            clu_centers.append(cluster_center)

        if len(clu_centers) != 0:  
            clu_centers = np.asarray(clu_centers)  # Nx2
            guidpost_centers[label_gd] = clu_centers

    # counts and std
    guidpost_eval = {}
    for label, pts in guidpost_centers.items():
        guidpost_eval[label] = [len(pts), np.linalg.norm(np.std(pts, axis=0))]

    return guidpost_centers, guidpost_eval


def get_anchor_pts_with_labels_outdoor(
    pts_src, label_src,
    pts_dst, label_dst,
    max_num_label,
    label_usefull,
    is_vis_guidpost=False,
    **args,
):
    label_usefull = np.asarray(label_usefull)
    # get guidpost
    guidpost_centers_src, guidpost_eval_src = extract_init_guidpost_outdoor(
        pts_src, label_src, is_vis=is_vis_guidpost, label_usefull=label_usefull)
    guidpost_centers_dst, guidpost_eval_dst = extract_init_guidpost_outdoor(
        pts_dst, label_dst, is_vis=is_vis_guidpost, label_usefull=label_usefull)

    # check None
    if np.any(np.asarray([guidpost_centers_src, guidpost_eval_src, guidpost_centers_dst, guidpost_eval_dst], dtype=object) == None):
        print("There are no guidport.")
        return False, None, None, None, None, None

    # guidpost to pts with label
    guidpost_pts_src = []
    guidpost_label_src = []
    guidpost_pts_dst = []
    guidpost_label_dst = []

    for label in guidpost_centers_src.keys():
        if label not in guidpost_centers_dst:
            continue

        if abs(guidpost_eval_src[label][0] - guidpost_eval_dst[label][0]) > 5:
            continue

        if (guidpost_eval_src[label][0] > 10 and guidpost_eval_src[label][1] <= 10) or (guidpost_eval_dst[label][0] > 10 and guidpost_eval_dst[label][1] <= 10):
            continue

        if len(guidpost_centers_src[label]) == 0 or len(guidpost_centers_dst[label]) == 0:
            continue

        # record
        guidpost_pts_src.append(guidpost_centers_src[label])
        guidpost_label_src.append(
            np.full(shape=(len(guidpost_pts_src[-1])), fill_value=label))
        guidpost_pts_dst.append(guidpost_centers_dst[label])
        guidpost_label_dst.append(
            np.full(shape=(len(guidpost_pts_dst[-1])), fill_value=label))

    is_use_gss = len(guidpost_pts_dst) > 0 and len(guidpost_pts_dst) > 0
    if not is_use_gss:
        return False, None, None, None, None, None

    # to numpy pts
    guidpost_pts_src = np.vstack(guidpost_pts_src)
    guidpost_label_src = np.concatenate(guidpost_label_src)
    guidpost_pts_dst = np.vstack(guidpost_pts_dst)
    guidpost_label_dst = np.concatenate(guidpost_label_dst)

    indic_list_src = []
    indic_list_dst = []
    for la in label_usefull:
        indic_src = guidpost_label_src == la
        indic_dst = guidpost_label_dst == la
        if not np.any(indic_src):  
            continue

        indic_list_src.append(indic_src)
        indic_list_dst.append(indic_dst)

        if len(indic_list_src) >= min(len(label_usefull), max_num_label): 
            break

    assert len(indic_list_src) == len(indic_list_dst)
    num_label_type = len(indic_list_dst) 
    for idx_indic in range(len(indic_list_src)):
        guidpost_label_src[indic_list_src[idx_indic]] = idx_indic
        guidpost_label_dst[indic_list_dst[idx_indic]] = idx_indic

    return (
        is_use_gss,
        guidpost_pts_src, guidpost_label_src,
        guidpost_pts_dst, guidpost_label_dst, num_label_type
    )


import taichi as ti
@ti.kernel
def taich_grab_gss(
    out: ti.types.ndarray(ndim=3, dtype=ti.uint8), 
    kpts: ti.types.ndarray(ndim=2, dtype=ti.float32), 
    # kpts: ti.types.ndarray(ndim=2, dtype=ti.float32), 
    # kpts: ti.Vector.field(n=3, dtype=ti.float32), 
    guidpost_pts: ti.types.ndarray(ndim=2, dtype=ti.float32), 
    guidpost_label: ti.types.ndarray(ndim=1, dtype=ti.int32),
    N: ti.int32, 
    L: ti.float32, 
):
    for i in ti.ndrange(kpts.shape[0]):
        for j in ti.ndrange(guidpost_pts.shape[0]):
            # distance 
            dis = ti.math.sqrt(
                    (kpts[i, 0] - guidpost_pts[j, 0]) * (kpts[i, 0] - guidpost_pts[j, 0]) + 
                    (kpts[i, 1] - guidpost_pts[j, 1]) * (kpts[i, 1] - guidpost_pts[j, 1]) + 
                    (kpts[i, 2] - guidpost_pts[j, 2]) * (kpts[i, 2] - guidpost_pts[j, 2]) 
            )

            idxs_ring: ti.int32 = dis / L

            # print(guidpost_label)
            # print(i, idxs_ring, guidpost_label[j], end="\t")
            
            if idxs_ring < N:
                out[i, idxs_ring, guidpost_label[j]] = ti.u8(1)


def grab_gss(
    kpts, guidpost_pts, guidpost_label,
    max_num_label,
    N,
    L,
    num_label_type,
    **args
):
    inner_rs = np.arange(N, dtype=np.float32) * L 
    max_radiu = inner_rs[-1] + L

    """
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.,
       26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50.,
       52., 54., 56., 58.], dtype=float32)

    max_radiu = 60
    """
    if not eval_taichi.is_use_taich:
        gss = np.full(shape=(
            len(kpts), N, num_label_type
        ), fill_value=False, dtype=np.ubyte)
        tree = KDTree(guidpost_pts)
        idxs, diss = tree.query_radius(
            kpts, r=max_radiu, return_distance=True)  
        for i in range(len(diss)):  
            idxs_ring = np.floor(diss[i] / L).astype(np.int32)

            for idx_r in range(N):
                if not np.any(idxs_ring == idx_r):  
                    continue

                indic = idxs_ring == idx_r
                label_seg = np.unique(
                    guidpost_label[idxs[i][indic]])  

                gss[i, idx_r, label_seg] = True

        return gss
    else:

        out = np.zeros(shape=(len(kpts), N, num_label_type), dtype=np.uint8)
        taich_grab_gss(out, kpts.astype(np.float32), guidpost_pts.astype(np.float32), guidpost_label.astype(np.int32), N, L)
    
        return out